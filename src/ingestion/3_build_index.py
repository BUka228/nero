import os
import json
import math
import torch
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# --- КОНФИГУРАЦИЯ ---
load_dotenv()

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/vector_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# Размер пачки. На M4 можно ставить побольше (256-512) для скорости.
BATCH_SIZE = 512 

class VectorIndexer:
    def __init__(self):
        print(f"🔧 Инициализация индексатора...")
        
        # 1. Определение устройства (Apple Silicon Optimization)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"⚡ Используемое устройство: {self.device.upper()}")

        # 2. Загрузка модели (один раз в память)
        print(f"📥 Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)

        # 3. Клиент базы данных
        print(f"💽 Подключение к ChromaDB: {CHROMA_DB_PATH}")
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

    def _format_passage(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.lstrip()
        if stripped.lower().startswith("passage:"):
            return text
        return f"passage: {text}"

    def _batch_generator(self, data: List[Any], batch_size: int):
        """
        Генератор, который возвращает:
        1. Пачку данных (batch)
        2. Глобальный индекс начала этой пачки (start_index)
        """
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size], i

    def index_chat_history(self):
        """Индексация переписки с сохранением глобального порядка"""
        input_file = PROCESSED_DIR / "chat_history_rag.json"
        
        if not input_file.exists():
            print(f"⚠️ Файл {input_file} не найден. Пропуск.")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        collection_name = "chat_history"
        print(f"\n📚 Индексация чата ({len(data)} сообщений) в '{collection_name}'...")
        
        # Удаляем старую коллекцию для чистоты эксперимента
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        
        # Создаем новую (cosine distance по умолчанию, для E5 подходит)
        collection = self.client.create_collection(name=collection_name)

        # Процесс по батчам
        total_batches = math.ceil(len(data) / BATCH_SIZE) if data else 0
        
        for batch, start_index in tqdm(self._batch_generator(data, BATCH_SIZE), total=total_batches):
            documents = []
            metadatas = []
            ids = []

            for i, item in enumerate(batch):
                # Вычисляем абсолютный номер сообщения в списке
                # Это КЛЮЧЕВОЙ момент для восстановления контекста
                global_idx = start_index + i
                
                # Формируем текст для поиска: "Роль: Текст"
                raw_text = f"{item['role']}: {item['content']}"
                text_content = self._format_passage(raw_text)
                
                documents.append(text_content)
                
                # Сохраняем global_index в метаданных
                metadatas.append({
                    "user_id": str(item.get("user_id", "")),
                    "role": item.get("role", "unknown"),
                    "timestamp": int(item.get("timestamp", 0)),
                    "date": item.get("date", ""),
                    "global_index": global_idx  # <--- ВОТ ОНО
                })
                
                # ID делаем уникальным (глобальный индекс)
                ids.append(str(global_idx))

            if documents:
                # Генерируем векторы
                embeddings = self.encoder.encode(
                    documents,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                
                # Пишем в базу
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )

        print(f"✅ Чат успешно проиндексирован.")

    def index_stickers(self):
        """Индексация стикеров (здесь порядок не важен, важен смысл)"""
        input_file = PROCESSED_DIR / "stickers_metadata.json"
        
        if not input_file.exists():
            print(f"⚠️ Файл {input_file} не найден. Пропуск.")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        collection_name = "sticker_search"
        print(f"\n🎨 Индексация стикеров ({len(data)} шт.) в '{collection_name}'...")

        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        
        collection = self.client.create_collection(name=collection_name)

        total_batches = math.ceil(len(data) / BATCH_SIZE) if data else 0

        for batch, _ in tqdm(self._batch_generator(data, BATCH_SIZE), total=total_batches):
            documents = []
            metadatas = []
            ids = []

            for item in batch:
                desc = item.get("description", "")
                if not desc:
                    continue

                text_content = self._format_passage(desc)

                documents.append(text_content)
                metadatas.append({
                    "path": item.get("path", ""),
                    "type": item.get("type", "static")
                })
                ids.append(item.get("path")) # ID = путь к файлу

            if documents:
                embeddings = self.encoder.encode(
                    documents,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
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