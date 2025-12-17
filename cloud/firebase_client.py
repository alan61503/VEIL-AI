import json
from pathlib import Path
from typing import Optional

from config import FIREBASE_CREDENTIALS, FIREBASE_CREDENTIALS_JSON

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError as exc:  # pragma: no cover - env guard
    firebase_admin = None
    credentials = firestore = None
    FIREBASE_IMPORT_ERROR = exc
else:
    FIREBASE_IMPORT_ERROR = None


_db: Optional["firestore.Client"] = None
_resolved_cred_path: Optional[Path] = None


def _resolve_credentials_file() -> Path:
    cred_path = Path(FIREBASE_CREDENTIALS)
    if cred_path.exists():
        return cred_path

    if FIREBASE_CREDENTIALS_JSON:
        global _resolved_cred_path
        if _resolved_cred_path and _resolved_cred_path.exists():
            return _resolved_cred_path

        try:
            parsed = json.loads(FIREBASE_CREDENTIALS_JSON)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("FIREBASE_CREDENTIALS_JSON is not valid JSON.") from exc

        cache_dir = Path(".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_dir / "firebase_credentials.json"
        temp_path.write_text(json.dumps(parsed))
        _resolved_cred_path = temp_path
        return temp_path

    raise FileNotFoundError(
        f"Firebase credentials file not found at {cred_path} and no FIREBASE_CREDENTIALS_JSON value provided."
    )


def init_firebase() -> "firestore.Client":
    """Initialise and return a singleton Firestore client."""
    global _db

    if _db:
        return _db

    if FIREBASE_IMPORT_ERROR:
        raise ImportError(
            "firebase_admin is required when CLOUD_PROVIDER=firebase. "
            "Install it with 'pip install firebase-admin'."
        ) from FIREBASE_IMPORT_ERROR

    cred_path = _resolve_credentials_file()

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)

    _db = firestore.client()
    return _db
