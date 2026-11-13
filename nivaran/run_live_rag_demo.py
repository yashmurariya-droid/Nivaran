"""End-to-end grounded RAG generation test for NIVARAN.

Executes exactly ONE live Gemini API call using gemini-3.5-flash-lite
with grounded ChromaDB context and applicant rejection notice.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from nivaran.ingestion import DocumentIngestionService
from nivaran.embeddings import EmbeddingService
from nivaran.vector_store import ChromaVectorStore
from nivaran.retrieval import RAGRetriever
from nivaran.generator import GeminiRAGGenerator


def run_e2e_rag_demo():
    print("=" * 70)
    print("NIVARAN END-TO-END GROUNDED RAG DEMO TEST")
    print("=" * 70)

    # 1. Load Applicant Rejection Document
    notice_path = Config.DATA_DIR / "raw" / "fictional_rejection_notice.pdf"
    if not notice_path.exists():
        raise FileNotFoundError(f"Applicant document not found at {notice_path}")

    ingestion = DocumentIngestionService()
    notice_chunks = ingestion.process_file(notice_path, document_type="rejection_notice")
    applicant_text = "\n\n".join([c.text for c in notice_chunks])
    print(f"\n[1] Loaded Applicant Document: {notice_path.name} ({len(notice_chunks)} chunks, {len(applicant_text)} chars)")

    # 2. Initialize Retrieval Pipeline
    vector_store = ChromaVectorStore(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )
    embedder = EmbeddingService(model_name=Config.EMBEDDING_MODEL_NAME)
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_service=embedder,
        default_top_k=3,
        similarity_threshold=0.25
    )

    # 3. Retrieve Evidence for the Exact Question
    exact_question = "Why was Aarav Sharma's fictional scholarship application rejected, and what should he correct?"
    print(f"\n[2] Exact Test Question:\n    \"{exact_question}\"")

    retrieved_chunks = retriever.retrieve(exact_question, top_k=3)
    print(f"\n[3] Retrieved Chunks from ChromaDB: {len(retrieved_chunks)}")

    # 4. Extract verified citation metadata directly from retrieval
    verified_citations = []
    for i, c in enumerate(retrieved_chunks, 1):
        meta = c.get("metadata", {})
        citation = {
            "index": i,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_id": c.get("chunk_id"),
            "similarity_score": c.get("similarity_score"),
            "snippet": c.get("text", "")[:180].replace("\n", " ")
        }
        verified_citations.append(citation)
        print(f"    - Source {i}: {citation['source']} (Page {citation['page']}) | Score: {citation['similarity_score']:.4f}")
        print(f"      Snippet: {citation['snippet']}...")

    formatted_context = retriever.format_context_for_prompt(retrieved_chunks)

    # 5. Call Gemini (Exactly ONE live request)
    # Ensure model is gemini-3.5-flash-lite as configured in .env
    model_name = Config.GEMINI_MODEL
    print(f"\n[4] Dispatching ONE live request to Gemini API (Model: {model_name})...")

    generator = GeminiRAGGenerator(
        api_key=Config.GEMINI_API_KEY,
        model_name=model_name
    )

    result = generator.generate_rag_answer(
        query=exact_question,
        context_text=formatted_context,
        applicant_document_text=applicant_text
    )

    print("\n" + "=" * 70)
    print("LIVE GENERATION OUTCOME")
    print("=" * 70)
    print(f"Success: {result['success']}")
    print(f"Model: {result['model']}")
    print(f"Error: {result['error']}")
    print("\n[5] Generated Answer:\n")
    print(result["answer"])

    print("\n[6] Verified Citations (Derived from ChromaDB Metadata):")
    for cit in verified_citations:
        print(f"    - [{cit['index']}] {cit['source']} (Page {cit['page']}) - Relevance: {cit['similarity_score']:.2%}")

    return {
        "success": result["success"],
        "model": result["model"],
        "answer": result["answer"],
        "citations": verified_citations,
    }


if __name__ == "__main__":
    run_e2e_rag_demo()
