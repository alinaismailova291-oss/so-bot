import os
import logging
import asyncio
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
import PyPDF2
from docx import Document
import chromadb
from chromadb.config import Settings
from snipsplit import Chunker

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Инициализация клиента для эмбеддингов и чата
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Настройка ChromaDB (локальное хранилище)
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))[citation:9]
collection = chroma_client.get_or_create_collection(name="sp_docs")

# Чанкер для разбиения текста (512 токенов, перекрытие 64)
chunker = Chunker(max_tokens=512, overlap_tokens=64)[citation:5]

# Счётчик документов для каждого чата
user_docs = {}

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
        "👋 Привет! Я бот для поиска по СП (RAG-система).\n\n"
        "Отправьте мне PDF или DOCX с последней редакцией. "
        "Я разобью его на части и запомню. Можно загрузить несколько документов.\n\n"
        "После загрузки задайте вопрос, и я найду ответ в загруженных документах."
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
            await update.message.reply_text("❌ Поддерживаются только PDF и DOCX.")
            return

        # Разбиваем текст на чанки
        chunks = [c.text for c in chunker.split(text) if len(c.text.strip()) > 20]
        if not chunks:
            await update.message.reply_text("❌ Не удалось извлечь текст из документа.")
            return

        await update.message.reply_text(f"📄 Разбиваю «{file_name}» на {len(chunks)} частей и создаю эмбеддинги...")

        # Генерируем эмбеддинги через OpenRouter (бесплатная модель)
        embeddings = []
        for i, chunk in enumerate(chunks):
            response = client.embeddings.create(
                model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
                input=chunk
            )
            embeddings.append(response.data[0].embedding)

            # Обновляем прогресс каждые 20 чанков
            if (i + 1) % 20 == 0:
                await update.message.reply_text(f"⏳ Обработано {i+1}/{len(chunks)} частей...")

        # Сохраняем в ChromaDB
        ids = [f"{chat_id}_{file_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name, "chat_id": str(chat_id)} for _ in chunks]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

        user_docs[chat_id] = user_docs.get(chat_id, 0) + 1
        await update.message.reply_text(f"✅ Документ «{file_name}» добавлен в базу знаний! Всего документов: {user_docs[chat_id]}")
    except Exception as e:
        logging.error(f"Ошибка при обработке файла: {e}")
        await update.message.reply_text("❌ Не удалось обработать файл. Попробуйте другой.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    question = update.message.text

    await update.message.reply_text("🤔 Ищу ответ в документах...")

    try:
        # Генерируем эмбеддинг для вопроса
        question_embedding = client.embeddings.create(
            model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
            input=question
        ).data[0].embedding

        # Ищем 5 самых похожих чанков в ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5,
            where={"chat_id": str(chat_id)}
        )

        if not results['documents'] or not results['documents'][0]:
            await update.message.reply_text("📭 В базе нет документов по этому чату. Сначала загрузите СП.")
            return

        # Собираем контекст
        context_text = "\n\n---\n\n".join(results['documents'][0])

        prompt = f"""Ты — эксперт по строительным нормативам (СП).
Ответь на вопрос, опираясь ТОЛЬКО на приведённые ниже фрагменты документов.
Если в них нет ответа, скажи об этом. В конце укажи, из каких источников взяты данные.

Контекст:
{context_text}

Вопрос: {question}
"""

        # Отправляем в ИИ
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        logging.error(f"Ошибка при обработке вопроса: {e}")
        await update.message.reply_text("❌ Ошибка при обработке вопроса. Попробуйте позже.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    application.run_polling(allowed_updates=Update.ALL_TYPES)[citation:11]

if __name__ == "__main__":
    main()
