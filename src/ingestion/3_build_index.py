import os
import json
import torch
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# --- КОНФИГУРАЦИЯ ---
load_dotenv()

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/vector_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# Настройки батчей (чем больше памяти, тем больше можно ставить)
BATCH_SIZE = 256 

class VectorIndexer:
    def __init__(self):
        print(f"🔧 Инициализация VectorIndexer...")
        
        # 1. Настройка устройства (M4 Optimization)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"⚡ Используемое устройство для эмбеддингов: {self.device.upper()}")

        # 2. Загрузка модели
        print(f"📥 Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)

        # 3. Подключение к ChromaDB
        print(f"💽 Подключение к базе данных: {CHROMA_DB_PATH}")
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

    def _batch_generator(self, data: List[Any], batch_size: int):
        """Генератор для разбиения данных на пачки"""
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def index_chat_history(self):
        """Индексация текстовой переписки"""
        input_file = PROCESSED_DIR / "chat_history_rag.json"
        
        if not input_file.exists():
            print(f"⚠️ Файл {input_file} не найден. Пропуск индексации чата.")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        collection_name = "chat_history"
        print(f"\n📚 Индексация истории чата ({len(data)} записей) в коллекцию '{collection_name}'...")
        
        # Пересоздаем коллекцию с нуля, чтобы не было дублей
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.client.create_collection(name=collection_name)

        # Процесс батчами
        for batch in tqdm(self._batch_generator(data, BATCH_SIZE), total=(len(data) // BATCH_SIZE) + 1):
            documents = []
            metadatas = []
            ids = []

            for idx, item in enumerate(batch):
                # Текст для векторизации
                # Для модели E5 часто добавляют префикс "passage: ", но для простоты RAG можно и без, 
                # если query тоже будет без префикса.
                # Для чатов важнее контекст: Кто сказал + Что сказал.
                text_content = f"{item['role']}: {item['content']}"
                
                documents.append(text_content)
                
                # Метаданные (чтобы потом фильтровать по дате или автору)
                metadatas.append({
                    "user_id": str(item.get("user_id", "")),
                    "role": item.get("role", "unknown"),
                    "timestamp": int(item.get("timestamp", 0)),
                    "date": item.get("date", "")
                })
                
                # Уникальный ID (можно timestamp + user_id, но проще uuid или просто инкремент глобальный)
                # Здесь используем timestamp как часть ID для уникальности внутри батча
                ids.append(f"{item.get('timestamp')}_{idx}")

            # Генерация эмбеддингов
            embeddings = self.encoder.encode(documents, convert_to_numpy=True, show_progress_bar=False)
            
            # Запись в базу
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        print(f"✅ Чат успешно проиндексирован.")

    def index_stickers(self):
        """Индексация описаний стикеров"""
        input_file = PROCESSED_DIR / "stickers_metadata.json"
        
        if not input_file.exists():
            print(f"⚠️ Файл {input_file} не найден. Пропуск индексации стикеров.")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        collection_name = "sticker_search"
        print(f"\n🎨 Индексация стикеров ({len(data)} записей) в коллекцию '{collection_name}'...")

        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.client.create_collection(name=collection_name)

        for batch in tqdm(self._batch_generator(data, BATCH_SIZE), total=(len(data) // BATCH_SIZE) + 1):
            documents = [] # Описания (то, что ищем)
            metadatas = [] # Пути к файлам (то, что возвращаем)
            ids = []

            for idx, item in enumerate(batch):
                desc = item.get("description", "")
                if not desc:
                    continue

                documents.append(desc)
                
                metadatas.append({
                    "path": item.get("path", ""),
                    "type": item.get("type", "static")
                })
                
                # ID на основе пути к файлу
                ids.append(str(item.get("path")))

            if not documents:
                continue

            embeddings = self.encoder.encode(documents, convert_to_numpy=True, show_progress_bar=False)
            
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        print(f"✅ Стикеры успешно проиндексированы.")

if __name__ == "__main__":
    indexer = VectorIndexer()
    indexer.index_chat_history()
    indexer.index_stickers()