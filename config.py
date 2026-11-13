import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file if present
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration supporting local and Vercel serverless environments."""

    # Serverless runtime detection
    IS_VERCEL = os.getenv("VERCEL") == "1"

    # Project Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"

    # In Vercel serverless functions, only /tmp is writable
    if IS_VERCEL:
        UPLOAD_FOLDER = Path("/tmp/uploads")
        CHROMA_PERSIST_DIRECTORY = Path("/tmp/chroma_db")
        LOGS_DIR = Path("/tmp/logs")
    else:
        UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "data/uploads")
        CHROMA_PERSIST_DIRECTORY = BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
        LOGS_DIR = BASE_DIR / "logs"

    # Server Configuration
    FLASK_ENV = os.getenv("FLASK_ENV", "production" if IS_VERCEL else "development")
    DEBUG = os.getenv("FLASK_DEBUG", "False" if IS_VERCEL else "True").lower() in ("true", "1", "t")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    SECRET_KEY = os.getenv("SECRET_KEY", "nivaran-secure-local-dev-fallback-key")

    # Upload Rules (Security)
    # Vercel serverless payload limit is 4.5 MB, local default is 16 MB
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "4" if IS_VERCEL else "16")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"pdf", "txt"}

    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    # Embeddings & ChromaDB:
    # On Vercel, default to Google's gemini-embedding-2 (3072 dim) to avoid large PyTorch bundle
    # In local development, defaults to all-MiniLM-L6-v2 unless overridden
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "gemini-embedding-2" if IS_VERCEL else "all-MiniLM-L6-v2"
    )
    CHROMA_COLLECTION_NAME = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "nivaran_scholarship_rules_gemini2" if (IS_VERCEL or "gemini" in os.getenv("EMBEDDING_MODEL_NAME", "").lower()) else "nivaran_scholarship_rules"
    )

    @classmethod
    def ensure_directories(cls):
        """Ensure all required runtime data directories exist."""
        cls.prepare_runtime_storage()

    @classmethod
    def prepare_runtime_storage(cls):
        """Prepare writable directories and hydrate ChromaDB storage on Vercel."""
        writable_dirs = [
            cls.UPLOAD_FOLDER,
            cls.LOGS_DIR,
        ]

        if cls.IS_VERCEL:
            writable_dirs.append(cls.CHROMA_PERSIST_DIRECTORY)
            # Hydrate /tmp/chroma_db from bundled data/chroma_db if not present
            source_chroma = cls.BASE_DIR / "data" / "chroma_db"
            if source_chroma.exists() and not (cls.CHROMA_PERSIST_DIRECTORY / "chroma.sqlite3").exists():
                try:
                    if cls.CHROMA_PERSIST_DIRECTORY.exists():
                        shutil.rmtree(cls.CHROMA_PERSIST_DIRECTORY)
                    shutil.copytree(source_chroma, cls.CHROMA_PERSIST_DIRECTORY)
                except Exception as e:
                    # Log or ignore if already prepared
                    pass
        else:
            writable_dirs.extend([
                cls.DATA_DIR,
                cls.KNOWLEDGE_BASE_DIR,
                cls.CHROMA_PERSIST_DIRECTORY,
            ])

        for path in writable_dirs:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                pass

    @classmethod
    def has_gemini_key(cls) -> bool:
        """Check whether a Gemini API key is configured."""
        return bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here")

    @classmethod
    def get_safe_summary(cls) -> dict:
        """Return non-sensitive config summary for diagnostic/health endpoints."""
        from nivaran.utils.logger import mask_sensitive_value

        return {
            "is_vercel": cls.IS_VERCEL,
            "flask_env": cls.FLASK_ENV,
            "host": cls.HOST,
            "port": cls.PORT,
            "gemini_model": cls.GEMINI_MODEL,
            "gemini_key_set": cls.has_gemini_key(),
            "gemini_key_preview": mask_sensitive_value(cls.GEMINI_API_KEY),
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "chroma_collection": cls.CHROMA_COLLECTION_NAME,
            "upload_folder": str(cls.UPLOAD_FOLDER),
            "chroma_dir": str(cls.CHROMA_PERSIST_DIRECTORY),
        }
