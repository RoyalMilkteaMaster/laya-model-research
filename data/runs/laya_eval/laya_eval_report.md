# Laya 多跳決策層評估報告

## 判定：PASS

- 產生時間：2026-09-29T18:30:37+00:00
- 模型目錄：`${PROJECT_RUNTIME_ROOT}/models/laya_multihop`
- 模型 hash（`laya_decision_client.model_hash`）：`3c0aba0a00c69fc8c739e959ee4fb77b3cace926866d2834f0e28455617592e7`
- 測試集：`${PROJECT_DATA_ROOT}/laya_decisions/test.jsonl`（sha256 `b8c19265547c91cc4d013da399cafa721372f9db764c486f34a3e6b5fbf43480`），決策 4032 筆、predict 1344 次
- 訓練耗時：10.8 min（NVIDIA GeForce RTX 4090），train 32451 筆、calibration 3996 筆
- 溫度 [choice, score, noul]：[5.0, 5.0, 5.0]
- 超參數：`{"epochs": 2, "micro_batch": 8, "grad_accum": 8, "group_size": 4, "lr_encoder": 2.5e-05, "lr_head": 0.0001, "weight_decay": 0.01, "sigma_start": 0.4, "sigma_end": 0.1, "w_sph": 0.75, "w_rps": 1.0, "ce_weight": 1.0, "max_len": 1024, "head_max_len": 256, "max_grad_norm": 1.0, "calibration_limit": null, "calibration_batch_size": 16, "seed": 42, "limit": null, "log_every": 20}`

## 門檻

| 檢查 | 數值 | 門檻 | 結果 |
|---|---|---|---|
| sufficient accuracy | 0.9271 | >= 0.75 | PASS |
| sufficient gain over zero-shot | 0.3356 | >= 0.2 | PASS |
| next_action accuracy | 0.8333 | >= 0.75 | PASS |
| next_action gain over zero-shot | 0.4859 | >= 0.2 | PASS |
| remaining_hops MAE | 0.2013 | <= 0.5 | PASS |

## 指標（微調 vs zero-shot 基底 `convaiinnovations/laya-multilingual`）

| 問題 | 模型 | n | accuracy | soft accuracy | Brier | ECE | MAE | MAE (argmax) | within-one |
|---|---|---|---|---|---|---|---|---|---|
| sufficient | 微調 | 1344 | 0.9271 | 0.9227 | 0.1343 | 0.0626 | — | — | — |
| sufficient | zero-shot | 1344 | 0.5915 | 0.5937 | 0.5551 | 0.2271 | — | — | — |
| next_action | 微調 | 1344 | 0.8333 | 0.8195 | 0.3024 | 0.1339 | — | — | — |
| next_action | zero-shot | 1344 | 0.3475 | 0.3461 | 0.7587 | 0.2119 | — | — | — |
| remaining_hops | 微調 | 1344 | 0.8177 | — | — | 0.1414 | 0.2013 | 0.1882 | 0.9740 |
| remaining_hops | zero-shot | 1344 | 0.3036 | — | — | 0.3750 | 0.8783 | 0.9055 | 0.6116 |

- 整體 accuracy：微調 0.8594、zero-shot 0.4142；整體 ECE：微調 0.1114、zero-shot 0.2703
- p50 延遲（一次 predict 回答同一 state 的三題）：微調 12.6 ms、zero-shot 13.2 ms
- MAE 用期望分數（同 `04_test_laya.py`，門檻依此）；MAE (argmax) 為 decision client 回傳的 `remaining` 誤差。Soft accuracy 與 Brier 只算 noul／choice。
