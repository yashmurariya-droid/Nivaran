"""Temporary server-side upload session store for NIVARAN.

Supports in-memory caching and authenticated AES-128-CBC + HMAC-SHA256 encrypted
stateless tokens via cryptography.fernet.Fernet with zlib compression.

Guarantees:
1. End-to-end confidentiality: Student document text and metadata are encrypted;
   they cannot be decoded or read by third parties from the token.
2. Authenticity & Integrity: Tampering with any byte of ciphertext invalidates the token.
3. Strict session isolation: Tokens are cryptographically bound to the user's session.
4. Auto-expiration: Enforces configurable TTL (default 2 hours).
5. Stateless continuity: Works across cold serverless function invocations without a database.
"""
import base64
import hashlib
import json
import time
import zlib
from threading import Lock
from typing import Dict, Any, Optional

from cryptography.fernet import Fernet, InvalidToken
from nivaran.utils.logger import get_logger

logger = get_logger("upload_session")

# Maximum permitted token payload size (1 MB) to stay far below Vercel's 4.5 MB body limit
MAX_TOKEN_LENGTH_BYTES = 1024 * 1024


class DocumentUploadStore:
    """Stateless & in-memory store for uploaded document texts using Fernet authenticated encryption."""

    def __init__(self, secret_key: str = "nivaran-secure-local-dev-fallback-key", ttl_seconds: int = 7200):
        self.secret_key = secret_key
        self.ttl_seconds = ttl_seconds
        # Derive standard 32-byte urlsafe base64 key from secret_key
        key_digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        self.fernet_key = base64.urlsafe_b64encode(key_digest)
        self.fernet = Fernet(self.fernet_key)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def store_document(
        self,
        session_id: str,
        filename: str,
        full_text: str,
        chunks_count: int,
    ) -> str:
        """Store extracted document text and return an authenticated, encrypted upload token."""
        if not session_id:
            raise ValueError("session_id is required to store document")

        now = time.time()
        safe_text = full_text or ""

        # Compress text with zlib level 9 and base85 encode for safe JSON payload
        compressed = zlib.compress(safe_text.encode("utf-8"), level=9)
        b85_str = base64.b85encode(compressed).decode("ascii")

        payload_dict = {
            "s": session_id,
            "f": filename,
            "t": b85_str,
            "c": chunks_count,
            "ts": now,
        }
        payload_bytes = json.dumps(payload_dict).encode("utf-8")

        # Encrypt with Fernet (AES-128-CBC + HMAC-SHA256 authenticated encryption)
        encrypted_token = self.fernet.encrypt(payload_bytes).decode("ascii")

        # Safety check against oversized payload
        if len(encrypted_token) > MAX_TOKEN_LENGTH_BYTES:
            raise ValueError(
                f"Extracted document text exceeds the maximum allowable token size "
                f"({len(encrypted_token)} bytes > {MAX_TOKEN_LENGTH_BYTES} bytes). "
                f"Please upload a smaller document."
            )

        # Also cache locally in memory for same-process fast access
        with self._lock:
            self._cleanup_expired(now)
            self._store[encrypted_token] = {
                "session_id": session_id,
                "filename": filename,
                "full_text": safe_text,
                "chunks_count": chunks_count,
                "created_at": now,
            }

        logger.info(
            "Stored document '%s' with encrypted upload_id (length: %d) for session '%s' (text chars: %d)",
            filename,
            len(encrypted_token),
            session_id[:8] + "...",
            len(safe_text),
        )
        return encrypted_token

    def get_document(self, upload_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored document text with strict session ownership verification.

        First checks the local memory cache. If not found (e.g. separate serverless
        function instance), decrypts, verifies, and unpacks the authenticated Fernet token.
        """
        if not upload_id or not session_id:
            return None

        now = time.time()

        # 1. Fast path: check in-memory store
        with self._lock:
            doc = self._store.get(upload_id)
            if doc:
                # Check expiration
                if now - doc["created_at"] > self.ttl_seconds:
                    logger.warning("Upload ID '%s' has expired in memory", upload_id[:16])
                    del self._store[upload_id]
                    return None

                # Strict ownership verification
                if doc["session_id"] != session_id:
                    logger.warning(
                        "Session mismatch for upload_id '%s': expected '%s...', got '%s...'",
                        upload_id[:16],
                        doc["session_id"][:8],
                        session_id[:8],
                    )
                    return None

                return doc

        # 2. Serverless fallback: decrypt and verify Fernet token
        try:
            decrypted_bytes = self.fernet.decrypt(
                upload_id.encode("ascii"),
                ttl=self.ttl_seconds
            )
            payload = json.loads(decrypted_bytes.decode("utf-8"))

            if not isinstance(payload, dict):
                logger.warning("Invalid token payload format")
                return None

            # Verify session ownership
            if payload.get("s") != session_id:
                logger.warning(
                    "Session mismatch in encrypted token: expected '%s...', got '%s...'",
                    session_id[:8],
                    str(payload.get("s"))[:8],
                )
                return None

            # Decompress text
            compressed_bytes = base64.b85decode(payload["t"].encode("ascii"))
            full_text = zlib.decompress(compressed_bytes).decode("utf-8")

            recovered_doc = {
                "session_id": payload["s"],
                "filename": payload.get("f", "document"),
                "full_text": full_text,
                "chunks_count": payload.get("c", 1),
                "created_at": payload.get("ts", now),
            }

            # Cache locally for future turns
            with self._lock:
                self._store[upload_id] = recovered_doc

            logger.info(
                "Successfully decrypted and recovered document '%s' from token (chars: %d)",
                recovered_doc["filename"],
                len(full_text),
            )
            return recovered_doc

        except InvalidToken:
            logger.warning("Upload token verification failed (invalid signature, tampered ciphertext, or expired TTL)")
            return None
        except Exception as e:
            logger.warning("Failed to unpack upload token: %s", str(e))
            return None

    def _cleanup_expired(self, now: float) -> None:
        """Purge entries older than TTL (must be called with lock held)."""
        expired = [
            uid for uid, doc in self._store.items()
            if now - doc["created_at"] > self.ttl_seconds
        ]
        for uid in expired:
            del self._store[uid]
        if expired:
            logger.debug("Purged %d expired upload entries from store", len(expired))
