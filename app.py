import os
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 16px;
    margin-bottom: 1.5rem; text-align: center; color: white;
}
.app-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.app-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0.5rem 0 0; }
.summary-box {
    background: #f8fafc; border-left: 4px solid #667eea;
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem;
    font-size: 0.9rem; line-height: 1.7; color: #374151;
}
.file-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0f4ff; color: #4338ca;
    border: 1px solid #c7d2fe; border-radius: 8px;
    padding: 4px 10px; font-size: 0.82rem; font-weight: 500; margin: 3px;
}
.source-tag {
    display: inline-block; background: #ede9fe; color: #5b21b6;
    border-radius: 6px; padding: 2px 8px;
    font-size: 0.78rem; font-weight: 500; margin: 2px;
}
.confidence-high { color: #059669; font-weight: 600; }
.confidence-mid  { color: #d97706; font-weight: 600; }
.confidence-low  { color: #dc2626; font-weight: 600; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


class DocumentStore:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.chunks = []
        self.index = None

    def add_document(self, text: str, filename: str) -> int:
        new_chunks = []
        chunk_size, overlap = 500, 50
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 100:
                new_chunks.append({"text": chunk, "source": filename})
        if not new_chunks:
            return 0
        embeddings = np.array(
            self.model.encode([c["text"] for c in new_chunks], show_progress_bar=False),
            dtype="float32"
        )
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> list:
        if self.index is None or not self.chunks:
            return []
        qv = np.array(self.model.encode([query]), dtype="float32")
        distances, indices = self.index.search(qv, top_k)
        max_d = max(distances[0]) if max(distances[0]) > 0 else 1
        return [
            {
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "confidence": round(max(0, 1 - distances[0][i] / max_d) * 100, 1)
            }
            for i, idx in enumerate(indices[0]) if idx < len(self.chunks)
        ]

    def clear(self):
        self.chunks = []
        self.index = None


class RAGChatbot:
    def __init__(self, groq_client, doc_store):
        self.client = groq_client
        self.doc_store = doc_store
        self.model = "llama-3.3-70b-versatile"

    def _call_llm(self, system: str, user: str, max_tokens=1500) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content

    def answer(self, question: str, chat_history: list) -> dict:
        chunks = self.doc_store.search(question, top_k=5)
        if not chunks:
            return {"answer": "Chưa có tài liệu. Vui lòng upload PDF trước.",
                    "sources": [], "avg_confidence": 0}
        context = "\n\n".join(
            f"[Nguồn: {c['source']} | Độ liên quan: {c['confidence']}%]\n{c['text']}"
            for c in chunks
        )
        system = f"""Bạn là trợ lý hỏi đáp tài liệu thông minh.
Trả lời DỰA TRÊN tài liệu, bằng tiếng Việt, rõ ràng có cấu trúc.
Nếu câu hỏi yêu cầu so sánh, phân tích từng tài liệu rồi so sánh.
Nếu không có trong tài liệu, nói rõ điều đó.

=== TÀI LIỆU ===
{context}"""
        messages = [{"role": "system", "content": system}]
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})
        res = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=1500
        )
        return {
            "answer": res.choices[0].message.content,
            "sources": list({c["source"] for c in chunks}),
            "avg_confidence": round(sum(c["confidence"] for c in chunks) / len(chunks), 1)
        }

    def summarize(self, text: str, filename: str) -> str:
        return self._call_llm(
            "Bạn là trợ lý tóm tắt tài liệu. Trả lời bằng tiếng Việt, súc tích.",
            f"Tóm tắt '{filename}' thành 4-6 ý chính ngắn gọn:\n\n{text[:4000]}",
            max_tokens=600
        )

    def suggest_questions(self, text: str) -> list:
        raw = self._call_llm(
            "Chỉ trả về đúng 3 câu hỏi, mỗi câu 1 dòng, không đánh số.",
            f"Tạo 3 câu hỏi thú vị nhất từ tài liệu:\n\n{text[:3000]}",
            max_tokens=200
        )
        return [q.strip() for q in raw.strip().split('\n') if q.strip()][:3]

    def compare(self, doc_names: list) -> str:
        return self.answer(
            f"So sánh chi tiết: {', '.join(doc_names)} về yêu cầu kỹ năng, quy trình, điều kiện.",
            []
        )["answer"]


@st.cache_resource
def load_models():
    return (
        SentenceTransformer("all-MiniLM-L6-v2"),
        Groq(api_key=os.getenv("GROQ_API_KEY"))
    )

def extract_text(f) -> str:
    return "\n\n".join(
        p.extract_text() for p in PdfReader(f).pages if p.extract_text()
    )

def confidence_color(s):
    return "confidence-high" if s >= 70 else "confidence-mid" if s >= 40 else "confidence-low"


def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="app-header">
        <h1>📄 PDF Chatbot</h1>
        <p>Hỏi đáp tài liệu thông minh · RAG + Vector Search + LLaMA 3.3</p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Đang khởi động AI..."):
        emb_model, groq_client = load_models()

    for key, val in [("doc_store", DocumentStore(emb_model)), ("chatbot", None),
                     ("chat_history", []), ("uploaded_files", {}),
                     ("suggestions", []), ("pending_question", None)]:
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.chatbot is None:
        st.session_state.chatbot = RAGChatbot(groq_client, st.session_state.doc_store)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("### 📁 Tài liệu")
        uploaded = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        if uploaded:
            for file in uploaded:
                if file.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Đang xử lý {file.name}..."):
                        text = extract_text(file)
                        st.session_state.doc_store.add_document(text, file.name)
                        summary = st.session_state.chatbot.summarize(text, file.name)
                        st.session_state.suggestions = st.session_state.chatbot.suggest_questions(text)
                        st.session_state.uploaded_files[file.name] = summary
                    st.success(f"✓ {file.name}")

        if st.session_state.uploaded_files:
            for name in st.session_state.uploaded_files:
                st.markdown(f'<span class="file-badge">📄 {name}</span>', unsafe_allow_html=True)

            st.divider()
            st.markdown("**📋 Tóm tắt tự động:**")
            for name, summary in st.session_state.uploaded_files.items():
                with st.expander(f"📄 {name}"):
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

            if st.session_state.suggestions:
                st.divider()
                st.markdown("**💡 Câu hỏi gợi ý:**")
                for q in st.session_state.suggestions:
                    if st.button(f"❓ {q}", key=f"sq_{q}", use_container_width=True):
                        st.session_state.pending_question = q
                        st.rerun()

            if len(st.session_state.uploaded_files) >= 2:
                st.divider()
                if st.button("⚖️ So sánh tất cả", use_container_width=True, type="primary"):
                    with st.spinner("Đang phân tích..."):
                        cmp = st.session_state.chatbot.compare(list(st.session_state.uploaded_files.keys()))
                    st.session_state.chat_history.append({"role": "user", "content": "So sánh tất cả tài liệu"})
                    st.session_state.chat_history.append({"role": "assistant", "content": cmp, "sources": list(st.session_state.uploaded_files.keys()), "avg_confidence": 0})
                    st.rerun()

            st.divider()
            if st.button("🗑️ Xóa tất cả", use_container_width=True):
                st.session_state.doc_store.clear()
                st.session_state.uploaded_files = {}
                st.session_state.chat_history = []
                st.session_state.suggestions = []
                st.rerun()

    with col_right:
        st.markdown("### 💬 Hỏi đáp")
        chat_container = st.container(height=480)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center;padding:3rem;color:#9ca3af;">
                    <div style="font-size:3rem">📄</div>
                    <div style="margin-top:0.5rem">Upload PDF bên trái để bắt đầu</div>
                </div>""", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        src_html = " ".join(f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"])
                        conf = msg.get("avg_confidence", 0)
                        st.markdown(f'{src_html} &nbsp; <span class="{confidence_color(conf)}">🎯 {conf}%</span>', unsafe_allow_html=True)

        if st.session_state.pending_question:
            q = st.session_state.pending_question
            st.session_state.pending_question = None
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Đang trả lời..."):
                result = st.session_state.chatbot.answer(q, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"], "avg_confidence": result["avg_confidence"]})
            st.rerun()

        question = st.chat_input("Hỏi về tài liệu...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Đang trả lời..."):
                result = st.session_state.chatbot.answer(question, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"], "avg_confidence": result["avg_confidence"]})
            st.rerun()


if __name__ == "__main__":
    main()