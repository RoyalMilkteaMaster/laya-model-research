"""讀取 multihop_benchmark/.env，解析三根目錄與 OLLAMA_HOST（ADR 0003）。

環境變數優先於 .env；程式碼不寫死任何根目錄路徑。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_DIR.parent
DEFAULT_ENV_FILE = PROJECT_DIR / ".env"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
REQUIRED_KEYS = ("PROJECT_DATA_ROOT", "PROJECT_RUNTIME_ROOT")


class SettingsError(RuntimeError):
    """設定缺漏或不合法。"""


@dataclass(frozen=True)
class Settings:
    data_root: Path
    runtime_root: Path
    ollama_host: str
    code_root: Path = CODE_ROOT

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def ollama_models_dir(self) -> Path:
        return self.runtime_root / "models" / "ollama"

    @property
    def hf_home(self) -> Path:
        return self.runtime_root / "cache" / "huggingface"

    @property
    def ollama_bin(self) -> Path:
        return self.runtime_root / "applications" / "ollama" / "bin" / "ollama"


def load_settings(
    env_file: Path | None = None, environ: Mapping[str, str] | None = None
) -> Settings:
    env_file = DEFAULT_ENV_FILE if env_file is None else env_file
    environ = os.environ if environ is None else environ

    values = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.is_file() else {}
    values.update({k: v for k, v in environ.items() if v})

    missing = [key for key in REQUIRED_KEYS if key not in values]
    if missing:
        raise SettingsError(
            f"缺少設定 {', '.join(missing)}：請設定環境變數，或依 "
            f"{PROJECT_DIR / '.env.example'} 建立 {env_file}"
        )

    roots = {}
    for key in REQUIRED_KEYS:
        path = Path(values[key]).expanduser()
        if not path.is_absolute():
            raise SettingsError(f"{key} 必須是絕對路徑，目前為 {values[key]!r}")
        roots[key] = path

    return Settings(
        data_root=roots["PROJECT_DATA_ROOT"],
        runtime_root=roots["PROJECT_RUNTIME_ROOT"],
        ollama_host=values.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/"),
    )
