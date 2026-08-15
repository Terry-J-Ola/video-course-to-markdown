import os
import pathlib
import re


KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHARED_API_KEY = "DASHSCOPE_API_KEY"
QWEN_API_KEY = "DASHSCOPE_QWEN_API_KEY"
ASR_API_KEY = "DASHSCOPE_ASR_API_KEY"


def get_qwen_api_key(environ: dict[str, str] | None = None) -> str | None:
    values = os.environ if environ is None else environ
    return values.get(QWEN_API_KEY) or values.get(SHARED_API_KEY)


def get_asr_api_key(environ: dict[str, str] | None = None) -> str | None:
    values = os.environ if environ is None else environ
    return values.get(ASR_API_KEY) or values.get(SHARED_API_KEY)


def configuration_status(source: str | None) -> dict[str, str | bool | None]:
    qwen_set = bool(get_qwen_api_key())
    asr_set = bool(get_asr_api_key())
    return {
        "source": source,
        "api_key_set": qwen_set and asr_set,
        "qwen_api_key_set": qwen_set,
        "asr_api_key_set": asr_set,
    }


def parse_env_file(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"Invalid .env key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_dashscope_environment(
    explicit: str | None,
    project_root: pathlib.Path,
    local_appdata: str | None = None,
) -> dict[str, str | bool | None]:
    if get_qwen_api_key() and get_asr_api_key():
        return configuration_status("environment")

    candidates: list[pathlib.Path] = []
    if explicit:
        explicit_path = pathlib.Path(explicit).expanduser().resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Explicit env file does not exist: {explicit_path}")
        candidates.append(explicit_path)
    else:
        candidates.append((project_root / ".env").resolve())
        base = local_appdata if local_appdata is not None else os.environ.get("LOCALAPPDATA")
        if base:
            candidates.append(
                (pathlib.Path(base) / "Codex" / "video-course-to-markdown" / ".env").resolve()
            )

    for path in candidates:
        if not path.is_file():
            continue
        values = parse_env_file(path)
        for key, value in values.items():
            os.environ.setdefault(key, value)
        return configuration_status(str(path))
    return configuration_status(None)
