import os
import logging
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
import PyPDF2
from docx import Document

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

documents = {}

def extract_text_from_pdf(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для поиска по СП.\n\n"
        "1. Отправь мне файл (PDF или DOCX) с последней редакцией.\n"
        "2. После загрузки задай любой вопрос по документу.\n\n"
        "Я отвечу, опираясь только на загруженный текст."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    file = await document.get_file()
    file_bytes = await file.download_as_bytearray()
    file_name = document.file_name

    try:
        if file_name.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif file_name.lower().endswith('.docx'):
            text = extract_text_from_docx(file_bytes)
        else:
            await update.message.reply_text("❌ Поддерживаются только PDF и DOCX.")
            return

        documents[update.effective_chat.id] = text
        await update.message.reply_text(
            f"✅ Документ «{file_name}» загружен! Теперь задайте вопрос."
        )
    except Exception as e:
        logging.error(f"Ошибка при обработке файла: {e}")
        await update.message.reply_text("❌ Не удалось обработать файл. Попробуйте другой.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    question = update.message.text

    if chat_id not in documents:
        await update.message.reply_text("⚠️ Сначала загрузите документ (PDF или DOCX).")
        return

    doc_text = documents[chat_id]
    max_chars = 12000
    if len(doc_text) > max_chars:
        doc_text = doc_text[:max_chars] + "\n...[текст обрезан]..."

    prompt = f"""Ты — помощник по строительным нормативам (СП).
Ответь на вопрос пользователя, опираясь ТОЛЬКО на следующий текст документа.
Если ответа в тексте нет, честно скажи об этом.

Текст документа:
{doc_text}

Вопрос: {question}
"""

    await update.message.reply_text("🤔 Думаю...")

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        await update.message.reply_text(answer)
    except Exception as e:
        logging.error(f"Ошибка при запросе к ИИ: {e}")
        await update.message.reply_text("❌ Ошибка при обращении к ИИ. Попробуйте позже.")

# Простой HTTP-сервер, чтобы Render видел открытый порт
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass  # не засоряем логи

def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()

def main():
    # Запускаем health-check сервер в отдельном потоке
    threading.Thread(target=run_health_server, daemon=True).start()

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
