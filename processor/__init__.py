"""Content Processing Service package.

Loads environment variables from a local .env file to support local
development and testing without external configuration.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _load_local_env():
    if load_dotenv is None:
        return
    # Try processor/.env first, then project root .env
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / ".env",
        pkg_dir.parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


_load_local_env()