import os
import logging
import asyncio
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
import PyPDF2
from docx import Document

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Получаем ключи из переменных окружения
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Инициализация клиента OpenRouter (совместим с OpenAI API)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Хранилище текста документов (в памяти, для простоты)
documents = {}

# Функция извлечения текста из PDF
def extract_text_from_pdf(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Функция извлечения текста из DOCX
def extract_text_from_docx(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для поиска по СП.\n\n"
        "1. Отправь мне файл (PDF или DOCX) с последней редакцией.\n"
        "2. После загрузки задай любой вопрос по документу.\n\n"
        "Я отвечу, опираясь только на загруженный текст."
    )

# Обработка загруженного документа
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

# Обработка текстовых вопросов
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

# Точка входа
async def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    # Держим бота запущенным
    stop_signal = asyncio.Event()
    await stop_signal.wait()

if __name__ == "__main__":
    asyncio.run(main())
