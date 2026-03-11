import os
from pathlib import Path


def _as_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_data_dir(project_root: Path, demo_mode: bool) -> Path:
    """
    Resolve runtime data directory in a team-safe way.

    Priority:
    1) IRIS_DATA_DIR (absolute or project-relative)
    2) Demo mode -> data/demo_guests (single canonical feedback/report path)
    3) IRIS_USE_REPO_DATA=true -> data/
    4) default isolated runtime dir -> runtime_data/
    """
    explicit_dir = str(os.environ.get("IRIS_DATA_DIR", "")).strip()
    if explicit_dir:
        candidate = Path(explicit_dir).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    if demo_mode:
        preferred_rel = Path("data/demo_guests")
    else:
        use_repo_data = _as_bool(os.environ.get("IRIS_USE_REPO_DATA", "false"))
        preferred_rel = Path("data") if use_repo_data else Path("runtime_data")

    preferred = project_root / preferred_rel
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = project_root / Path("runtime_data")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
