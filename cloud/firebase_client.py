from pathlib import Path
from typing import Optional

from config import FIREBASE_CREDENTIALS

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

    cred_path = Path(FIREBASE_CREDENTIALS)
    if not cred_path.exists():
        raise FileNotFoundError(
            f"Firebase credentials file not found at {cred_path}."
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)

    _db = firestore.client()
    return _db
