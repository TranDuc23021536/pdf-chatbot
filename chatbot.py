import os
from groq import Groq
from pypdf import PdfReader
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()

console = Console()


class PDFChatbot:
    """
    Chatbot hỏi đáp tài liệu PDF sử dụng Groq AI.
    Đây là ứng dụng RAG (Retrieval-Augmented Generation) đơn giản:
    - Retrieval: đọc và lưu nội dung PDF
    - Augmented Generation: đưa nội dung đó vào prompt cho AI trả lời
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Không tìm thấy GROQ_API_KEY trong file .env")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.pdf_content = ""
        self.pdf_name = ""
        self.chat_history = []

    def load_pdf(self, pdf_path: str) -> bool:
        """Đọc và extract toàn bộ text từ file PDF"""
        try:
            reader = PdfReader(pdf_path)
            text_pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_pages.append(f"[Trang {i+1}]\n{text}")

            self.pdf_content = "\n\n".join(text_pages)
            self.pdf_name = os.path.basename(pdf_path)
            self.chat_history = []
            return True

        except FileNotFoundError:
            console.print(f"[red]Không tìm thấy file: {pdf_path}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Lỗi khi đọc PDF: {e}[/red]")
            return False

    def ask(self, question: str) -> str:
        """Gửi câu hỏi kèm nội dung PDF cho AI trả lời"""
        if not self.pdf_content:
            return "Bạn chưa load file PDF nào. Dùng lệnh 'load <đường_dẫn_file.pdf>'"

        system_prompt = f"""Bạn là trợ lý hỏi đáp tài liệu. Hãy trả lời câu hỏi DỰA TRÊN nội dung tài liệu được cung cấp.
Nếu thông tin không có trong tài liệu, hãy nói rõ "Tài liệu không đề cập đến vấn đề này."
Trả lời bằng tiếng Việt, ngắn gọn và chính xác.

=== NỘI DUNG TÀI LIỆU: {self.pdf_name} ===
{self.pdf_content[:6000]}"""

        messages = [{"role": "system", "content": system_prompt}]

        for item in self.chat_history[-3:]:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["answer"]})

        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            self.chat_history.append({"question": question, "answer": answer})
            return answer

        except Exception as e:
            return f"Lỗi khi gọi AI: {e}"

    def summarize(self) -> str:
        """Tóm tắt toàn bộ nội dung PDF"""
        if not self.pdf_content:
            return "Bạn chưa load file PDF nào."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Hãy tóm tắt tài liệu sau thành các ý chính, trình bày rõ ràng bằng tiếng Việt:\n\n{self.pdf_content[:6000]}"
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Lỗi: {e}"


def print_welcome():
    console.print(Panel.fit(
        "[bold]PDF Chatbot[/bold] — Hỏi đáp tài liệu bằng AI\n\n"
        "[dim]Lệnh:[/dim]\n"
        "  [cyan]load <file.pdf>[/cyan]  — Load file PDF\n"
        "  [cyan]summary[/cyan]          — Tóm tắt tài liệu\n"
        "  [cyan]history[/cyan]          — Xem lịch sử hỏi đáp\n"
        "  [cyan]clear[/cyan]            — Xóa lịch sử\n"
        "  [cyan]exit[/cyan]             — Thoát\n\n"
        "[dim]Hoặc gõ câu hỏi bất kỳ để hỏi về tài liệu[/dim]",
        title="Chào mừng",
        border_style="blue"
    ))


def main():
    print_welcome()
    bot = PDFChatbot()

    while True:
        try:
            user_input = console.input("\n[bold cyan]Bạn:[/bold cyan] ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                console.print("[dim]Tạm biệt![/dim]")
                break

            elif user_input.lower().startswith("load "):
                pdf_path = user_input[5:].strip()
                with console.status(f"Đang đọc {pdf_path}..."):
                    success = bot.load_pdf(pdf_path)
                if success:
                    pages = len(PdfReader(pdf_path).pages)
                    console.print(Panel(
                        f"[green]✓ Đã load:[/green] {bot.pdf_name}\n"
                        f"[dim]Số trang: {pages} | Sẵn sàng trả lời câu hỏi[/dim]",
                        border_style="green"
                    ))

            elif user_input.lower() == "summary":
                with console.status("Đang tóm tắt tài liệu..."):
                    result = bot.summarize()
                console.print(Panel(result, title="Tóm tắt tài liệu", border_style="yellow"))

            elif user_input.lower() == "history":
                if not bot.chat_history:
                    console.print("[dim]Chưa có lịch sử hội thoại[/dim]")
                else:
                    for i, item in enumerate(bot.chat_history, 1):
                        console.print(f"\n[bold]{i}. Hỏi:[/bold] {item['question']}")
                        console.print(f"   [dim]Đáp:[/dim] {item['answer'][:150]}...")

            elif user_input.lower() == "clear":
                bot.chat_history = []
                console.print("[dim]Đã xóa lịch sử hội thoại[/dim]")

            else:
                with console.status("Đang suy nghĩ..."):
                    answer = bot.ask(user_input)
                console.print(Panel(answer, title="Bot", border_style="blue"))

        except KeyboardInterrupt:
            console.print("\n[dim]Tạm biệt![/dim]")
            break


if __name__ == "__main__":
    main()