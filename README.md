\# PDF Chatbot — Hỏi đáp tài liệu bằng AI



Chatbot cho phép upload file PDF bất kỳ và hỏi đáp trực tiếp dựa trên nội dung tài liệu, sử dụng kỹ thuật RAG (Retrieval-Augmented Generation).



\## Tính năng

\- Load file PDF và extract toàn bộ nội dung

\- Hỏi đáp tự nhiên bằng tiếng Việt dựa trên tài liệu

\- Tóm tắt tài liệu tự động

\- Lưu lịch sử hội thoại trong phiên làm việc



\## Công nghệ sử dụng

\- Python 3.11

\- Groq API (LLaMA 3.3 70B)

\- pypdf — đọc file PDF

\- rich — giao diện terminal

\- python-dotenv — quản lý biến môi trường

\- OOP (class-based architecture)



\## Kiến trúc RAG

PDF file → pypdf extract text → đưa vào system prompt → Groq LLM → câu trả lời

Thay vì để AI tự trả lời từ kiến thức có sẵn, chatbot cung cấp nội dung tài liệu thực tế vào mỗi prompt — đảm bảo câu trả lời chính xác và có căn cứ.



\## Cài đặt



```bash

git clone https://github.com/your-username/pdf-chatbot

cd pdf-chatbot

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



Tạo file `.env`:

GROQ\_API\_KEY=your\_key\_here



\## Chạy chương trình



```bash

python chatbot.py

```



\## Ví dụ sử dụng

Bạn: load tailieu.pdf

✓ Đã load: tailieu.pdf | Số trang: 5

Bạn: tài liệu này nói về gì?

Bot: Tài liệu mô tả chương trình thực tập tại...

Bạn: summary

Bot: Các ý chính của tài liệu...

