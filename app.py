"""NIVARAN Flask Application Foundation.

Provides secure HTTP endpoints, error handling, health checks,
and integration with the RAG pipeline.
"""
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from nivaran.utils.logger import get_logger
from nivaran.ingestion import DocumentIngestionService
from nivaran.embeddings import EmbeddingService
from nivaran.vector_store import ChromaVectorStore
from nivaran.retrieval import RAGRetriever
from nivaran.generator import GeminiRAGGenerator
from nivaran.upload_session import DocumentUploadStore

logger = get_logger("app")


def create_app(config_class=Config) -> Flask:
    """Application factory for NIVARAN."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    app.config.from_object(config_class)

    # Ensure runtime directories exist
    config_class.ensure_directories()

    # Initialize Services
    ingestion_service = DocumentIngestionService()
    embedding_service = EmbeddingService(
        model_name=config_class.EMBEDDING_MODEL_NAME,
        api_key=config_class.GEMINI_API_KEY,
    )
    vector_store = ChromaVectorStore(
        persist_directory=config_class.CHROMA_PERSIST_DIRECTORY,
        collection_name=config_class.CHROMA_COLLECTION_NAME,
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )
    generator = GeminiRAGGenerator(
        api_key=config_class.GEMINI_API_KEY,
        model_name=config_class.GEMINI_MODEL,
    )
    upload_store = DocumentUploadStore(
        secret_key=config_class.SECRET_KEY,
        ttl_seconds=7200,
    )

    # Attach services to app instance for route access
    app.extensions["nivaran"] = {
        "ingestion": ingestion_service,
        "embeddings": embedding_service,
        "vector_store": vector_store,
        "retriever": retriever,
        "generator": generator,
        "upload_store": upload_store,
        "config": config_class,
    }

    # Security Headers
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # Global Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not Found", "message": "Endpoint does not exist"}), 404
        return render_template("index.html"), 404

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(error):
        return jsonify({
            "error": "File Too Large",
            "message": f"Maximum allowed file size is {config_class.MAX_CONTENT_LENGTH // (1024 * 1024)}MB."
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error("Internal Server Error: %s", str(error))
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred."}), 500

    # Frontend UI Routes (Four Civic Hearth Screens & Welcome Landing)
    @app.route("/")
    @app.route("/welcome")
    @app.route("/analysis")
    @app.route("/document-check")
    @app.route("/ask")
    @app.route("/ask-questions")
    @app.route("/steps")
    @app.route("/next-steps")
    def index():
        return render_template("index.html")

    # Health & System Status Endpoint
    @app.route("/api/health", methods=["GET"])
    def health_check():
        summary = config_class.get_safe_summary()
        summary.update({
            "status": "healthy",
            "vector_store_docs_count": vector_store.count(),
            "gemini_ready": generator.is_configured(),
        })
        return jsonify(summary), 200

    # Knowledge Base Stats
    @app.route("/api/knowledge-base/stats", methods=["GET"])
    def kb_stats():
        return jsonify({
            "collection_name": config_class.CHROMA_COLLECTION_NAME,
            "total_chunks": vector_store.count(),
            "embedding_model": config_class.EMBEDDING_MODEL_NAME,
        }), 200

    # Upload Applicant Documents (Foundation)
    @app.route("/api/upload", methods=["POST"])
    def upload_document():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        safe_filename_str = secure_filename(file.filename)
        if not safe_filename_str:
            return jsonify({"error": "Invalid filename"}), 400

        ext = Path(safe_filename_str).suffix.lower().lstrip(".")
        if ext not in config_class.ALLOWED_EXTENSIONS:
            return jsonify({
                "error": "Unsupported file type",
                "message": f"Allowed extensions: {', '.join(config_class.ALLOWED_EXTENSIONS)}"
            }), 400

        # Save to private upload folder (never public static directory)
        save_path = config_class.UPLOAD_FOLDER / safe_filename_str
        file.save(save_path)
        logger.info("Securely saved uploaded file: %s (Size: %d bytes)", safe_filename_str, save_path.stat().st_size)

        # Ingestion test on uploaded file
        try:
            chunks = ingestion_service.process_file(save_path)
            full_text = "\n\n".join([c.text for c in chunks])
            extracted_preview = " ".join([c.text for c in chunks[:2]])[:500] if chunks else ""

            # Ensure user session ID exists for isolation
            user_sid = session.get("user_sid")
            if not user_sid:
                user_sid = str(uuid.uuid4())
                session["user_sid"] = user_sid

            upload_id = upload_store.store_document(
                session_id=user_sid,
                filename=safe_filename_str,
                full_text=full_text,
                chunks_count=len(chunks),
            )
            session["current_upload_id"] = upload_id

            return jsonify({
                "success": True,
                "upload_id": upload_id,
                "filename": safe_filename_str,
                "chunks_extracted": len(chunks),
                "preview": extracted_preview,
            }), 200
        except ValueError as e:
            logger.warning("Upload validation failed for %s: %s", safe_filename_str, str(e))
            return jsonify({
                "error": "Document Too Large",
                "message": str(e)
            }), 413
        except Exception as e:
            logger.error("Failed to parse uploaded document %s: %s", safe_filename_str, str(e))
            return jsonify({
                "error": "Document Parsing Failed",
                "message": str(e)
            }), 500

    # Document-Driven Defect Analysis Endpoint
    @app.route("/api/analyze", methods=["POST"])
    def analyze_document():
        data = request.get_json(silent=True) or {}
        upload_id = data.get("upload_id", "").strip()
        raw_lang = data.get("language", "en")
        clean_lang = "en"
        if isinstance(raw_lang, str):
            lowered = raw_lang.lower().strip()
            if lowered in ("hi", "hindi"):
                clean_lang = "hi"
            elif lowered in ("hinglish", "hi-en"):
                clean_lang = "hinglish"

        if not upload_id:
            return jsonify({"error": "Bad Request", "message": "upload_id is required"}), 400

        user_sid = session.get("user_sid")
        if not user_sid:
            return jsonify({"error": "Forbidden", "message": "Access denied: No active session found"}), 403

        # Strict session ownership check
        stored_doc = upload_store.get_document(upload_id=upload_id, session_id=user_sid)
        if not stored_doc:
            return jsonify({
                "error": "Not Found",
                "message": "Upload not found or access denied for this session"
            }), 404

        full_doc_text = stored_doc["full_text"]

        # 1. Retrieve policy evidence from ChromaDB
        analysis_query = (
            "What grounds for rejection, defect codes, certificate validity rules, "
            "and remediation procedures apply to this application?"
        )
        retrieved_chunks = retriever.retrieve(analysis_query, top_k=4)
        formatted_context = retriever.format_context_for_prompt(retrieved_chunks)

        # 2. Execute structured defect analysis with Gemini
        analysis_result = generator.analyze_document_defects(
            document_text=full_doc_text,
            context_text=formatted_context,
            language=clean_lang,
        )

        # 3. Format citations derived exclusively from retrieval metadata
        citations = [
            {
                "source": c.get("metadata", {}).get("source"),
                "page": c.get("metadata", {}).get("page"),
                "chunk_id": c.get("chunk_id"),
                "similarity_score": c.get("similarity_score"),
                "snippet": c.get("text", "")[:200]
            }
            for c in retrieved_chunks
        ]

        status_code = 200
        if not analysis_result.get("success"):
            if analysis_result.get("error") == "MALFORMED_ANALYSIS_OUTPUT":
                status_code = 502
            else:
                status_code = 500

        return jsonify({
            "success": analysis_result["success"],
            "error": analysis_result.get("error"),
            "message": analysis_result.get("message"),
            "language": clean_lang,
            "upload_id": upload_id,
            "filename": stored_doc["filename"],
            "applicant_name": analysis_result.get("applicant_name"),
            "document_reference": analysis_result.get("document_reference"),
            "document_status": analysis_result.get("document_status"),
            "summary": analysis_result.get("summary"),
            "findings": analysis_result.get("findings", []),
            "evidence_gaps": analysis_result.get("evidence_gaps", []),
            "citations": citations,
            "grounding_audit": analysis_result.get("grounding_audit"),
        }), status_code

    # RAG Query Endpoint (Foundation)
    @app.route("/api/query", methods=["POST"])
    def query_rag():
        data = request.get_json(silent=True) or {}
        query_text = data.get("query", "").strip()
        raw_lang = data.get("language", "en")
        clean_lang = "en"
        if isinstance(raw_lang, str):
            lowered = raw_lang.lower().strip()
            if lowered in ("hi", "hindi"):
                clean_lang = "hi"
            elif lowered in ("hinglish", "hi-en"):
                clean_lang = "hinglish"

        if not query_text:
            return jsonify({"error": "Query cannot be empty"}), 400

        applicant_text = data.get("applicant_document_text", "")

        # 1. Retrieve relevant chunks from vector store
        retrieved_chunks = retriever.retrieve(query_text, top_k=4)
        formatted_context = retriever.format_context_for_prompt(retrieved_chunks)

        # 2. Generate answer with Gemini
        generation_result = generator.generate_rag_answer(
            query=query_text,
            context_text=formatted_context,
            applicant_document_text=applicant_text,
            language=clean_lang,
        )

        return jsonify({
            "query": query_text,
            "answer": generation_result["answer"],
            "model": generation_result["model"],
            "language": clean_lang,
            "citations": [
                {
                    "source": c.get("metadata", {}).get("source"),
                    "page": c.get("metadata", {}).get("page"),
                    "chunk_id": c.get("chunk_id"),
                    "similarity_score": c.get("similarity_score"),
                    "snippet": c.get("text", "")[:200]
                }
                for c in retrieved_chunks
            ],
            "grounding_audit": generation_result.get("grounding_audit"),
            "success": generation_result["success"],
            "error": generation_result["error"]
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    logger.info("Starting NIVARAN on http://%s:%d", Config.HOST, Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
