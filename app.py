import os
import logging
import threading
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
import PyPDF2
from docx import Document
import numpy as np

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

# Хранилище: {chat_id: [(chunk_text, embedding, source_name), ...]}
store = {}


def split_text(text, chunk_size=700, overlap=100):
    """Простое разбиение текста на чанки."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_text_from_pdf(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])


def get_embedding(text):
    response = client.embeddings.create(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        input=text[:2000]
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для поиска по СП.\n\n"
        "Отправьте PDF или DOCX. Можно загрузить несколько документов.\n\n"
        "Команды:\n/list — список документов\n/clear — очистить всё"
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    chat_id = update.effective_chat.id
    file = await document.get_file()
    file_bytes = await file.download_as_bytearray()
    file_name = document.file_name

    try:
        if file_name.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif file_name.lower().endswith('.docx'):
            text = extract_text_from_docx(file_bytes)
        else:
            await update.message.reply_text("❌ Только PDF и DOCX.")
            return

        chunks = split_text(text)
        if not chunks:
            await update.message.reply_text("❌ Не удалось извлечь текст.")
            return

        await update.message.reply_text(f"📄 Обрабатываю «{file_name}» — {len(chunks)} частей...")

        if chat_id not in store:
            store[chat_id] = []

        success_count = 0
        for i, chunk in enumerate(chunks):
            try:
                emb = get_embedding(chunk)
                store[chat_id].append((chunk, emb, file_name))
                success_count += 1
            except Exception as e:
                logging.error(f"Ошибка на чанке {i}: {e}")
                # Пропускаем проблемный чанк, продолжаем
                continue

            # Пауза между запросами, чтобы не упереться в rate limit
            await asyncio.sleep(1.5)

            if (i + 1) % 20 == 0:
                await update.message.reply_text(f"⏳ {i+1}/{len(chunks)}...")

        total = len(store[chat_id])
        await update.message.reply_text(
            f"✅ «{file_name}» загружен.\n"
            f"Обработано частей: {success_count}/{len(chunks)}\n"
            f"Всего частей в базе: {total}"
        )
    except Exception as e:
        logging.error(f"Ошибка файла: {e}")
        await update.message.reply_text("❌ Ошибка при обработке файла.")


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    question = update.message.text

    if chat_id not in store or not store[chat_id]:
        await update.message.reply_text("📭 Сначала загрузите документ.")
        return

    await update.message.reply_text("🤔 Ищу ответ...")

    try:
        q_emb = get_embedding(question)

        scored = []
        for chunk, emb, source in store[chat_id]:
            sim = cosine_similarity(q_emb, emb)
            scored.append((sim, chunk, source))
        scored.sort(reverse=True, key=lambda x: x[0])

        top = scored[:5]
        context_text = "\n\n---\n\n".join(
            f"[{src}]\n{chunk}" for _, chunk, src in top
        )

        prompt = f"""Ты эксперт по строительным нормативам (СП).
Ответь на вопрос, опираясь ТОЛЬКО на фрагменты ниже.
Если ответа нет — скажи об этом.

Фрагменты:
{context_text}

Вопрос: {question}
"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        await update.message.reply_text(response.choices[0].message.content)

    except Exception as e:
        logging.error(f"Ошибка вопроса: {e}")
        await update.message.reply_text("❌ Ошибка при обработке вопроса.")


async def list_docs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in store or not store[chat_id]:
        await update.message.reply_text("📭 Пусто.")
        return
    sources = sorted(set(item[2] for item in store[chat_id]))
    names = "\n".join(f"• {s}" for s in sources)
    await update.message.reply_text(f"📚 Документов ({len(sources)}):\n{names}")


async def clear_docs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    store[update.effective_chat.id] = []
    await update.message.reply_text("🗑 Очищено.")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass


def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


def main():
    threading.Thread(target=run_health_server, daemon=True).start()
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("list", list_docs))
    application.add_handler(CommandHandler("clear", clear_docs))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
