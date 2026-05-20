import os
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ==================== CLASS CORE ====================

class DocumentStore:
    """
    Lưu trữ và tìm kiếm tài liệu bằng Vector Search.
    Đây là phần Retrieval trong RAG:
    - Mỗi đoạn văn được chuyển thành vector (embedding)
    - Khi có câu hỏi, tìm các đoạn văn có vector gần nhất
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.chunks = []       # lưu text gốc
        self.index = None      # FAISS index để tìm kiếm vector

    def add_document(self, text: str, filename: str):
        """Chia PDF thành chunks, tạo embedding, lưu vào FAISS"""
        # Chia text thành từng đoạn ~500 ký tự, overlap 50 ký tự
        new_chunks = []
        chunk_size = 500
        overlap = 50
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 100:
                new_chunks.append({"text": chunk, "source": filename})

        if not new_chunks:
            return 0

        # Tạo embedding cho từng chunk
        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        # Thêm vào FAISS index
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, top_k: int = 4) -> list:
        """Tìm các đoạn văn liên quan nhất đến câu hỏi"""
        if self.index is None or len(self.chunks) == 0:
            return []

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(distances[0][i])
                })
        return results

    def clear(self):
        self.chunks = []
        self.index = None


class RAGChatbot:
    """
    Chatbot RAG kết hợp Vector Search + LLM:
    1. Retrieval: dùng DocumentStore tìm đoạn văn liên quan
    2. Augmented Generation: đưa đoạn văn đó vào prompt cho LLM trả lời
    """

    def __init__(self, groq_client: Groq, doc_store: DocumentStore):
        self.client = groq_client
        self.doc_store = doc_store
        self.model = "llama-3.3-70b-versatile"

    def answer(self, question: str, chat_history: list) -> str:
        # Bước 1: Retrieval — tìm đoạn văn liên quan
        relevant_chunks = self.doc_store.search(question, top_k=4)

        if not relevant_chunks:
            return "Chưa có tài liệu nào được upload. Vui lòng upload file PDF trước."

        # Bước 2: Augmented Generation — ghép context vào prompt
        context = "\n\n".join([
            f"[Nguồn: {c['source']}]\n{c['text']}"
            for c in relevant_chunks
        ])

        system_prompt = f"""Bạn là trợ lý hỏi đáp tài liệu thông minh. 
Hãy trả lời câu hỏi DỰA TRÊN các đoạn tài liệu được cung cấp.
Nếu thông tin không có trong tài liệu, hãy nói rõ điều đó.
Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc.
Cuối câu trả lời, ghi rõ nguồn tài liệu bạn dựa vào.

=== TÀI LIỆU LIÊN QUAN ===
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ==================== STREAMLIT UI ====================

@st.cache_resource
def load_models():
    """Load model 1 lần duy nhất, cache lại"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedding_model, groq_client


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 PDF Chatbot")
    st.caption("Hỏi đáp tài liệu bằng AI — RAG + Vector Search + LLaMA 3.3")

    # Load models
    with st.spinner("Đang khởi động AI models..."):
        embedding_model, groq_client = load_models()

    # Khởi tạo session state
    if "doc_store" not in st.session_state:
        st.session_state.doc_store = DocumentStore(embedding_model)
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(groq_client, st.session_state.doc_store)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Layout 2 cột
    col_left, col_right = st.columns([1, 2])

    # ---- CỘT TRÁI: Upload PDF ----
    with col_left:
        st.subheader("📁 Tài liệu")

        uploaded = st.file_uploader(
            "Upload file PDF",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded:
            for file in uploaded:
                if file.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Đang xử lý {file.name}..."):
                        text = extract_pdf_text(file)
                        n_chunks = st.session_state.doc_store.add_document(text, file.name)
                        st.session_state.uploaded_files.append(file.name)
                    st.success(f"✓ {file.name} ({n_chunks} chunks)")

        if st.session_state.uploaded_files:
            st.divider()
            st.markdown("**Tài liệu đã load:**")
            for name in st.session_state.uploaded_files:
                st.markdown(f"- 📄 {name}")

            if st.button("🗑️ Xóa tất cả", use_container_width=True):
                st.session_state.doc_store.clear()
                st.session_state.uploaded_files = []
                st.session_state.chat_history = []
                st.rerun()

    # ---- CỘT PHẢI: Chat ----
    with col_right:
        st.subheader("💬 Hỏi đáp")

        # Hiển thị lịch sử chat
        chat_container = st.container(height=450)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("Upload PDF bên trái rồi đặt câu hỏi bên dưới!")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input câu hỏi
        question = st.chat_input("Hỏi về tài liệu...")
        if question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.spinner("Đang tìm kiếm và trả lời..."):
                answer = st.session_state.chatbot.answer(
                    question,
                    st.session_state.chat_history
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            st.rerun()


if __name__ == "__main__":
    main()