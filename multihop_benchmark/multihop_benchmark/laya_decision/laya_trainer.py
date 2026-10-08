"""Laya RLCD 微調與 held-out 溫度校準（改自 `laya_zh_tw/03_train_laya.py`）。

流程與教學版相同：`laya.load(基底)` 取 tokenizer／model／cfg → gradient checkpointing → 每個 micro batch
做 Gaussian exploration（群組 `group_size`）→ `laya.common.proper_reward`（log + spherical + score 題的 RPS）→
GRPO-style group baseline 與標準化 advantage → policy-gradient loss + `ce_weight` × soft cross-entropy →
`grad_accum` 次累積後 clip、AdamW（encoder／head 分開學習率）、cosine 排程；sigma 依 epoch 由 `sigma_start`
線性降到 `sigma_end`。訓練後在 calibration 集依型別（choice、score、noul）以 LBFGS 擬合溫度。

與教學版的差異：
- 超參數集中於 `TrainConfig`，CLI 可覆寫；預設 micro batch 8 × grad accum 8（effective 64，教學版 T4 為 2 × 32）。
- 支援 bf16 的 GPU 用 bf16 autocast（基底 cfg 的 `amp_dtype` 為 bf16），不需 GradScaler；否則 fp16 + GradScaler。
- calibration 預設用全部列（教學版取前 600）；`limit` 只供冒煙，限制 train／calibration 讀入列數。
- 輸出先寫到 `<output_dir>.partial`，以 `laya.load()` 驗證可載入後才換入 `output_dir`（`publish_model`；這一步也讓
  laya 先修正 `tokenizer/tokenizer_config.json`，之後目錄內容與 `model_hash` 不再變動）；驗證或換入失敗時保留原模型。
- 訓練進度寫 `train_laya.jsonl`（`train_progress`、`train_epoch_finished`、`calibration_finished`、`model_saved`）。

輸出目錄：`model.safetensors`（fp16）、`encoder/`、`tokenizer/`、`rl_agent_config.json`（`temperature` =
[choice, score, noul]，`max_len` 1024、`head_max_len` 256，`training` 記錄本次微調資訊）。
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_MODEL = "convaiinnovations/laya-multilingual"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    micro_batch: int = 8
    grad_accum: int = 8
    group_size: int = 4
    lr_encoder: float = 2.5e-5
    lr_head: float = 1.0e-4
    weight_decay: float = 0.01
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    ce_weight: float = 1.0
    max_len: int = 1024
    head_max_len: int = 256
    max_grad_norm: float = 1.0
    calibration_limit: int | None = None
    calibration_batch_size: int = 16
    seed: int = 42
    limit: int | None = None
    log_every: int = 20

    def validate(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size 至少要 2，否則 group baseline 沒有比較對象")
        if self.sigma_start <= 0 or self.sigma_end <= 0:
            raise ValueError("sigma_start 與 sigma_end 都必須大於 0")
        for name in ("epochs", "micro_batch", "grad_accum", "calibration_batch_size", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必須 ≥ 1：{getattr(self, name)}")


def _read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_item(row: Mapping[str, Any], tokenizer: Any, *, max_len: int, head_max_len: int) -> dict[str, Any] | None:
    """一列決策 → laya 訓練 item；選項數與 target 長度不符（或 marker 被截掉）時回傳 None。"""
    from laya.common import QTYPES, build_sequence, render_options

    q = {"t": row["type"], "ins": row["instructions"], "crit": row.get("criteria")}
    ids, markers = build_sequence(tokenizer, row["state"], q, max_len=max_len, head_max_len=head_max_len)
    expected = len(render_options(q))
    if len(markers) != expected or len(row["target"]) != expected:
        return None
    return {"ids": ids, "markers": markers, "qtype": QTYPES[row["type"]], "target": list(row["target"]),
            "label": int(np.argmax(row["target"]))}


def build_items(
    path: str | Path, tokenizer: Any, *, max_len: int, head_max_len: int, limit: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    items, skipped = [], 0
    for row in _read_jsonl(path, limit):
        item = build_item(row, tokenizer, max_len=max_len, head_max_len=head_max_len)
        if item is None:
            skipped += 1
        else:
            items.append(item)
    return items, skipped


def fit_temperature(samples: list[tuple[Any, Any]]) -> float:
    """以 held-out (logits, target) 擬合單一溫度（LBFGS 最小化 soft CE），夾在 [0.5, 5]；少於 10 筆回傳 1.0。"""
    import torch

    if len(samples) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in samples)
    logits = torch.full((len(samples), kmax), -1e4, dtype=torch.float32)
    targets = torch.zeros((len(samples), kmax), dtype=torch.float32)
    for i, (z, target) in enumerate(samples):
        logits[i, : len(z)] = torch.as_tensor(np.asarray(z), dtype=torch.float32)
        targets[i, : len(target)] = torch.as_tensor(np.asarray(target), dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def _save_model(model: Any, tokenizer: Any, cfg: Mapping[str, Any], out_dir: Path) -> None:
    import torch
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        state[name] = tensor.half() if torch.is_floating_point(tensor) else tensor
    save_file(state, str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    (out_dir / "rl_agent_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_with_laya(model_dir: Path) -> None:
    import laya

    agent = laya.load(str(model_dir), device="cpu")
    del agent


def publish_model(
    partial: str | Path, output_dir: str | Path, *, validate: Callable[[Path], None] = _validate_with_laya,
) -> None:
    """先以 `validate(partial)`（預設 `laya.load`）驗證暫存目錄，通過才換入 `output_dir`。

    驗證失敗時刪除暫存目錄、正式目錄維持原樣；換入時改名失敗則把舊模型改回原位。正式路徑因此只會出現
    驗證過的模型（`laya.load` 對 tokenizer_config 的修正也發生在暫存目錄，換入後 `model_hash` 不再變動）。
    """
    partial, output_dir = Path(partial), Path(output_dir)
    try:
        validate(partial)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    old = output_dir.with_name(output_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    had_previous = output_dir.exists()
    if had_previous:
        output_dir.rename(old)
    try:
        partial.rename(output_dir)
    except BaseException:
        if had_previous:
            old.rename(output_dir)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if had_previous:
        shutil.rmtree(old)


def train(
    train_path: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig = TrainConfig(),
    *,
    base_model: str = BASE_MODEL,
    device: str = "cuda",
    logger: Any = None,
    progress: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """RLCD 微調 + 溫度校準，輸出 `laya.load()` 可載入的目錄；回傳訓練摘要（耗時、曲線、超參數、筆數、溫度）。"""
    config.validate()
    import torch
    from contextlib import nullcontext

    import laya
    from laya.common import collate_items, proper_reward

    def log(message: str, event: str, **fields: Any) -> None:
        progress(message)
        if logger is not None:
            logger.info(message, event=event, **fields)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU；train-laya 需要 GPU")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    dev = torch.device(device)
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    def amp():
        return torch.autocast("cuda", dtype=amp_dtype) if dev.type == "cuda" else nullcontext()

    started = time.time()
    base_agent = laya.load(base_model, device="cpu")
    tokenizer, model, cfg = base_agent.tok, base_agent.model, dict(base_agent.cfg)
    del base_agent
    cfg["max_len"], cfg["head_max_len"], cfg["gradient_checkpointing"] = config.max_len, config.head_max_len, True

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        try:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.encoder.gradient_checkpointing_enable()
    if hasattr(model.encoder.config, "use_cache"):
        model.encoder.config.use_cache = False
    model.head_checkpointing = True
    model.to(dev).train()

    lengths = {"max_len": config.max_len, "head_max_len": config.head_max_len}
    train_items, skipped_train = build_items(train_path, tokenizer, limit=config.limit, **lengths)
    calibration_items, skipped_cal = build_items(calibration_path, tokenizer, limit=config.limit, **lengths)
    if config.calibration_limit is not None:
        calibration_items = calibration_items[: config.calibration_limit]
    if not train_items or not calibration_items:
        raise RuntimeError("沒有可用的訓練或校準資料；請先執行 make-laya-data")

    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": config.lr_encoder}, {"params": head_params, "lr": config.lr_head}],
        weight_decay=config.weight_decay,
    )
    micro_batches_per_epoch = math.ceil(len(train_items) / config.micro_batch)
    updates_per_epoch = math.ceil(micro_batches_per_epoch / config.grad_accum)
    total_updates = max(1, config.epochs * updates_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and not use_bf16)
    log(
        f"train-laya 開始：train={len(train_items)}（略過 {skipped_train}）calibration={len(calibration_items)}"
        f"（略過 {skipped_cal}）effective batch={config.micro_batch * config.grad_accum} updates={total_updates}"
        f" amp={'bf16' if use_bf16 else 'fp16'}",
        "train_started", hyperparameters=asdict(config), train_items=len(train_items),
        calibration_items=len(calibration_items), total_updates=total_updates,
    )

    def move(batch):
        for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target"):
            batch[key] = batch[key].to(dev)
        return batch

    train_started = time.time()
    global_update = 0
    epochs_summary, curve = [], []
    window_loss = window_reward = 0.0
    window_count = 0
    for epoch in range(config.epochs):
        random.Random(config.seed + epoch).shuffle(train_items)
        micro_batches = [train_items[i : i + config.micro_batch] for i in range(0, len(train_items), config.micro_batch)]
        sigma = config.sigma_start + (config.sigma_end - config.sigma_start) * (epoch / max(1, config.epochs - 1))
        epoch_loss = epoch_reward = 0.0
        seen = 0
        for group_start in range(0, len(micro_batches), config.grad_accum):
            accumulation = micro_batches[group_start : group_start + config.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            for chunk in accumulation:
                batch = move(collate_items([chunk], tokenizer.pad_token_id))
                with amp():
                    logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                      batch["marker_mask"], batch["qtype"])
                logits = logits.float()  # RLCD 計算用 float32
                mask, target, qtype = batch["marker_mask"], batch["target"], batch["qtype"]
                option_count = mask.sum(-1, keepdim=True).float().clamp_min(1.0)

                # 1. Gaussian exploration（零和擾動，只在有效選項上）
                eps = torch.randn((config.group_size,) + tuple(logits.shape), device=dev) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / option_count) * mask
                sampled = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(sampled.masked_fill(~mask, -1e4), dim=-1)
                # 2. proper scoring reward + GRPO-style group baseline
                with torch.no_grad():
                    reward = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=config.w_sph, w_rps=config.w_rps)
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage = advantage / (advantage.std(unbiased=False) + 1e-6)
                # 3. policy gradient
                logp = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
                loss_rl = -(advantage * logp).mean()
                # 4. soft cross-entropy guidance
                loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)).sum(-1).mean()

                full_loss = loss_rl + config.ce_weight * loss_ce
                scaler.scale(full_loss / len(accumulation)).backward()
                loss_value, reward_value = float(full_loss.detach()), float(reward.mean())
                epoch_loss += loss_value
                epoch_reward += reward_value
                window_loss += loss_value
                window_reward += reward_value
                window_count += 1
                seen += 1

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_update += 1
            if global_update % config.log_every == 0 or global_update == total_updates:
                point = {"update": global_update, "epoch": epoch + 1, "loss": window_loss / window_count,
                         "reward": window_reward / window_count, "sigma": sigma,
                         "elapsed_s": round(time.time() - train_started, 1)}
                curve.append(point)
                window_loss = window_reward = 0.0
                window_count = 0
                log(f"epoch={epoch + 1}/{config.epochs} update={global_update}/{total_updates} "
                    f"loss={point['loss']:.4f} reward={point['reward']:.3f} sigma={sigma:.3f} "
                    f"elapsed={point['elapsed_s']:.0f}s", "train_progress", **point)
        epochs_summary.append({"epoch": epoch + 1, "mean_loss": epoch_loss / max(1, seen),
                               "mean_reward": epoch_reward / max(1, seen), "sigma": sigma})
        log(f"epoch {epoch + 1} 完成：loss={epochs_summary[-1]['mean_loss']:.4f} "
            f"reward={epochs_summary[-1]['mean_reward']:.3f}", "train_epoch_finished", **epochs_summary[-1])
    train_elapsed = time.time() - train_started

    model.eval()
    by_type: dict[int, list[tuple[Any, Any]]] = {0: [], 1: [], 2: []}
    with torch.inference_mode():
        for start in range(0, len(calibration_items), config.calibration_batch_size):
            chunk = calibration_items[start : start + config.calibration_batch_size]
            batch = move(collate_items([chunk], tokenizer.pad_token_id))
            with amp():
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                  batch["marker_mask"], batch["qtype"])
            logits = logits.float().cpu().numpy()
            for i, item in enumerate(chunk):
                by_type[item["qtype"]].append((logits[i, : len(item["markers"])], item["target"]))
    temperatures = [fit_temperature(by_type[i]) for i in range(3)]  # QTYPES 順序：choice, score, noul
    log(f"溫度校準完成 [choice, score, noul] = {[round(t, 3) for t in temperatures]}", "calibration_finished",
        temperature=temperatures, calibration_items=len(calibration_items))

    output_dir = Path(output_dir)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    peak_vram_mb = round(torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else None
    cfg["temperature"] = temperatures
    cfg.pop("temperature_by_options", None)
    cfg["training"] = {
        "fine_tuned_from_checkpoint": base_model, "method": "RLCD (laya_zh_tw/03_train_laya.py)",
        "updates": global_update, "epochs_completed": config.epochs, "hours": round(train_elapsed / 3600, 3),
        "train_items": len(train_items), "calibration_items": len(calibration_items), "seed": config.seed,
    }
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    _save_model(model, tokenizer, cfg, partial)
    del model, optimizer
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    publish_model(partial, output_dir)
    elapsed = time.time() - started

    from multihop_benchmark.laya_decision.laya_decision_client import model_hash

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": base_model,
        "output_dir": str(output_dir),
        "model_hash": model_hash(output_dir),
        "train_path": str(train_path),
        "calibration_path": str(calibration_path),
        "train_items": len(train_items),
        "train_skipped": skipped_train,
        "calibration_items": len(calibration_items),
        "calibration_skipped": skipped_cal,
        "hyperparameters": asdict(config),
        "effective_batch": config.micro_batch * config.grad_accum,
        "amp_dtype": "bf16" if use_bf16 else "fp16",
        "updates": global_update,
        "elapsed_s": round(elapsed, 1),
        "train_elapsed_s": round(train_elapsed, 1),
        "gpu": gpu,
        "peak_vram_mb": peak_vram_mb,
        "torch": torch.__version__,
        "laya": laya.__version__,
        "temperature": temperatures,
        "epochs": epochs_summary,
        "curve": curve,
    }
    log(f"模型已寫入 {output_dir}（總耗時 {elapsed / 60:.1f} min）", "model_saved", output_dir=str(output_dir),
        model_hash=summary["model_hash"], elapsed_s=summary["elapsed_s"], temperature=temperatures)
    return summary
