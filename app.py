import os
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

load_dotenv()


# ==================== CLASSES ====================

class DocumentStore:
    """
    Lưu trữ và tìm kiếm tài liệu bằng Vector Search.
    RAG - Retrieval: chuyển text thành vector, tìm đoạn liên quan nhất.
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.chunks = []
        self.index = None

    def add_document(self, text: str, filename: str) -> int:
        new_chunks = []
        chunk_size = 500
        overlap = 50
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 100:
                new_chunks.append({"text": chunk, "source": filename})

        if not new_chunks:
            return 0

        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> list:
        if self.index is None or len(self.chunks) == 0:
            return []

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        max_dist = max(distances[0]) if max(distances[0]) > 0 else 1
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                confidence = max(0, 1 - distances[0][i] / max_dist)
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "confidence": round(confidence * 100, 1)
                })
        return results

    def clear(self):
        self.chunks = []
        self.index = None


class RAGChatbot:
    """
    RAG Chatbot: Vector Search + LLM
    1. Retrieval: tìm đoạn văn liên quan bằng FAISS
    2. Augmented Generation: đưa context vào prompt cho LLaMA trả lời
    """

    def __init__(self, groq_client: Groq, doc_store: DocumentStore):
        self.client = groq_client
        self.doc_store = doc_store
        self.model = "llama-3.3-70b-versatile"

    def answer(self, question: str, chat_history: list) -> dict:
        relevant_chunks = self.doc_store.search(question, top_k=5)

        if not relevant_chunks:
            return {
                "answer": "Chưa có tài liệu nào. Vui lòng upload PDF trước.",
                "sources": [],
                "avg_confidence": 0
            }

        context = "\n\n".join([
            f"[Nguồn: {c['source']} | Độ liên quan: {c['confidence']}%]\n{c['text']}"
            for c in relevant_chunks
        ])

        sources = list({c["source"] for c in relevant_chunks})
        avg_confidence = round(sum(c["confidence"] for c in relevant_chunks) / len(relevant_chunks), 1)

        system_prompt = f"""Bạn là trợ lý hỏi đáp tài liệu thông minh.
Trả lời câu hỏi DỰA TRÊN tài liệu được cung cấp, bằng tiếng Việt, rõ ràng có cấu trúc.
Nếu câu hỏi yêu cầu SO SÁNH nhiều tài liệu, hãy phân tích từng tài liệu rồi so sánh.
Nếu thông tin không có trong tài liệu, nói rõ điều đó.

=== TÀI LIỆU ===
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1500,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "avg_confidence": avg_confidence
        }

    def compare_documents(self, doc_names: list) -> str:
        """So sánh trực tiếp nhiều tài liệu với nhau"""
        question = f"Hãy so sánh chi tiết các tài liệu sau về: yêu cầu kỹ năng, quy trình, điều kiện và điểm khác biệt nổi bật: {', '.join(doc_names)}"
        result = self.answer(question, [])
        return result["answer"]


class ReportGenerator:
    """Tạo báo cáo PDF từ lịch sử hội thoại"""

   @staticmethod
    def generate_pdf(chat_history: list, doc_names: list) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', fontSize=16,
                                     spaceAfter=12, fontName='Helvetica-Bold')
        heading_style = ParagraphStyle('CustomHeading', fontSize=12,
                                       spaceAfter=6, spaceBefore=12,
                                       fontName='Helvetica-Bold')
        body_style = ParagraphStyle('CustomBody', fontSize=10,
                                    spaceAfter=6, leading=14,
                                    fontName='Helvetica')
        meta_style = ParagraphStyle('CustomMeta', fontSize=9,
                                    textColor='grey', spaceAfter=4,
                                    fontName='Helvetica')
        story = []
        story.append(Paragraph("PDF Chatbot - Bao Cao Hoi Dap", title_style))
        story.append(Paragraph(
            f"Tao luc: {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
            f"Tai lieu: {', '.join(doc_names) if doc_names else 'Khong co'}",
            meta_style
        ))
        story.append(Spacer(1, 0.5*cm))
        history = [m for m in chat_history if m["role"] in ("user", "assistant")]
        qa_pairs = []
        i = 0
        while i < len(history) - 1:
            if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
                qa_pairs.append((history[i]["content"], history[i+1]["content"]))
                i += 2
            else:
                i += 1
        for idx, (q, a) in enumerate(qa_pairs, 1):
            safe_q = q.encode('ascii', 'ignore').decode('ascii')
            safe_a = a.encode('ascii', 'ignore').decode('ascii')
            safe_q = safe_q.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')
            safe_a = safe_a.replace('<','&lt;').replace('>','&gt;').replace('&','&amp;')
            story.append(Paragraph(f"Cau {idx}: {safe_q}", heading_style))
            for line in safe_a.split('\n'):
                if line.strip():
                    story.append(Paragraph(line.strip(), body_style))
            story.append(Spacer(1, 0.3*cm))
        doc.build(story)
        return buffer.getvalue()

# ==================== STREAMLIT UI ====================

@st.cache_resource
def load_models():
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
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

    st.title("📄 PDF Chatbot")
    st.caption("RAG + Vector Search + LLaMA 3.3 — So sánh tài liệu & Xuất báo cáo")

    with st.spinner("Đang khởi động AI..."):
        embedding_model, groq_client = load_models()

    if "doc_store" not in st.session_state:
        st.session_state.doc_store = DocumentStore(embedding_model)
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(groq_client, st.session_state.doc_store)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    col_left, col_right = st.columns([1, 2])

    # ---- CỘT TRÁI ----
    with col_left:
        st.subheader("📁 Tài liệu")

        uploaded = st.file_uploader("Upload PDF (nhiều file)", type="pdf",
                                     accept_multiple_files=True)
        if uploaded:
            for file in uploaded:
                if file.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Đang xử lý {file.name}..."):
                        text = extract_pdf_text(file)
                        n = st.session_state.doc_store.add_document(text, file.name)
                        st.session_state.uploaded_files.append(file.name)
                    st.success(f"✓ {file.name} ({n} chunks)")

        if st.session_state.uploaded_files:
            st.divider()
            st.markdown("**Đã load:**")
            for name in st.session_state.uploaded_files:
                st.markdown(f"- 📄 {name}")

            # Nút so sánh
            if len(st.session_state.uploaded_files) >= 2:
                st.divider()
                st.markdown("**So sánh tài liệu**")
                if st.button("⚖️ So sánh tất cả tài liệu", use_container_width=True):
                    with st.spinner("Đang phân tích và so sánh..."):
                        comparison = st.session_state.chatbot.compare_documents(
                            st.session_state.uploaded_files
                        )
                    st.session_state.chat_history.append(
                        {"role": "user", "content": f"So sánh các tài liệu: {', '.join(st.session_state.uploaded_files)}"}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": comparison}
                    )
                    st.rerun()

            st.divider()

            # Xuất báo cáo PDF
            if st.session_state.chat_history:
                if st.button("📥 Xuất báo cáo PDF", use_container_width=True):
                    with st.spinner("Đang tạo báo cáo..."):
                        pdf_bytes = ReportGenerator.generate_pdf(
                            st.session_state.chat_history,
                            st.session_state.uploaded_files
                        )
                    st.download_button(
                        label="⬇️ Tải báo cáo",
                        data=pdf_bytes,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            if st.button("🗑️ Xóa tất cả", use_container_width=True):
                st.session_state.doc_store.clear()
                st.session_state.uploaded_files = []
                st.session_state.chat_history = []
                st.rerun()

    # ---- CỘT PHẢI ----
    with col_right:
        st.subheader("💬 Hỏi đáp")

        chat_container = st.container(height=450)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("Upload PDF bên trái rồi đặt câu hỏi!")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "sources" in msg:
                        cols = st.columns(2)
                        with cols[0]:
                            st.caption(f"📚 Nguồn: {', '.join(msg['sources'])}")
                        with cols[1]:
                            st.caption(f"🎯 Độ liên quan: {msg['avg_confidence']}%")

        question = st.chat_input("Hỏi về tài liệu hoặc so sánh nhiều tài liệu...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Đang tìm kiếm và trả lời..."):
                result = st.session_state.chatbot.answer(
                    question, st.session_state.chat_history
                )

            assistant_msg = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "avg_confidence": result["avg_confidence"]
            }
            st.session_state.chat_history.append(assistant_msg)
            st.rerun()


if __name__ == "__main__":
    main()