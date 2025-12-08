import os
import json
import torch
import chromadb
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from dateparser.search import search_dates

load_dotenv()

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/vector_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# --- НАСТРОЙКИ ---
SEARCH_LIMIT = 30           
MERGE_GAP_THRESHOLD = 20    
PADDING = 5                 
STICKER_THRESHOLD = 0.6
# НОВОЕ: Максимальный разрыв времени внутри одного блока контекста (в секундах)
# 43200 секунд = 12 часов. Если разрыв больше, контекст обрывается.
MAX_TIME_GAP = 43200 

class RagRetriever:
    def __init__(self):
        print("🔍 Инициализация SMART RAG движка...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        try:
            self.chat_collection = self.client.get_collection("chat_history")
            self.sticker_collection = self.client.get_collection("sticker_search")
        except Exception as e:
            print(f"❌ Ошибка загрузки коллекций: {e}")
            raise

        history_path = PROCESSED_DIR / "chat_history_rag.json"
        with open(history_path, 'r', encoding='utf-8') as f:
            self.full_history = json.load(f)
        
        self.history_len = len(self.full_history)
        print(f"✅ История загружена: {self.history_len} сообщений.")

    def _extract_date_bounds(self, query: str) -> Optional[Tuple[float, float]]:
        """Возвращает start_ts и end_ts если найдена дата"""
        dates = search_dates(query, languages=['ru', 'en'])
        if not dates: return None
        
        found_text, date_obj = dates[0]
        start_of_day = date_obj.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end_of_day = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        
        print(f"📅 Режим даты: {date_obj.date()}")
        return (start_of_day, end_of_day)

    def search_context(self, query: str) -> str:
        if not query: return ""

        # 1. Проверка даты
        date_bounds = self._extract_date_bounds(query)
        date_filter = None
        
        if date_bounds:
            start_ts, end_ts = date_bounds
            date_filter = {
                "$and": [
                    {"timestamp": {"$gte": start_ts}},
                    {"timestamp": {"$lte": end_ts}}
                ]
            }

        # 2. Поиск векторов
        query_vec = self.encoder.encode([query], convert_to_numpy=True).tolist()
        
        try:
            results = self.chat_collection.query(
                query_embeddings=query_vec,
                n_results=SEARCH_LIMIT,
                where=date_filter # Фильтр на уровне поиска кандидатов
            )
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return ""

        hit_indices = set()
        if results['metadatas'] and results['metadatas'][0]:
            for meta in results['metadatas'][0]:
                idx = meta.get('global_index')
                if idx is not None: hit_indices.add(idx)

        if not hit_indices: return ""

        # 3. Склейка островов
        sorted_hits = sorted(list(hit_indices))
        merged_blocks = []
        
        if sorted_hits:
            current_start = max(0, sorted_hits[0] - PADDING)
            current_end = min(self.history_len, sorted_hits[0] + PADDING)

            for i in range(1, len(sorted_hits)):
                next_hit = sorted_hits[i]
                if next_hit <= (current_end + MERGE_GAP_THRESHOLD):
                    new_end = min(self.history_len, next_hit + PADDING)
                    current_end = max(current_end, new_end)
                else:
                    merged_blocks.append((current_start, current_end))
                    current_start = max(0, next_hit - PADDING)
                    current_end = min(self.history_len, next_hit + PADDING)
            merged_blocks.append((current_start, current_end))

        # 4. Формирование с УМНОЙ ФИЛЬТРАЦИЕЙ
        final_output = []
        
        for start, end in merged_blocks:
            # Срезаем блок
            safe_end = min(end + 1, self.history_len)
            chunk_indices = range(start, safe_end)
            
            valid_msgs = []
            last_ts = 0

            for idx in chunk_indices:
                msg = self.full_history[idx]
                curr_ts = int(msg.get('timestamp', 0))

                # --- ПРОВЕРКА 1: ДАТА (Если включен режим даты) ---
                if date_bounds:
                    # Если мы ищем конкретную дату, МЫ НЕ ДОЛЖНЫ показывать сообщения из других дней,
                    # даже если это "контекст".
                    if not (date_bounds[0] <= curr_ts <= date_bounds[1]):
                        continue

                # --- ПРОВЕРКА 2: ВРЕМЕННОЙ РАЗРЫВ (Time Gap) ---
                # Если это не первое сообщение в блоке и разрыв больше 12 часов
                if last_ts > 0 and (curr_ts - last_ts) > MAX_TIME_GAP:
                    # Если разрыв случился внутри блока склейки — добавляем разделитель
                    if valid_msgs:
                        valid_msgs.append("--- [ПРОШЛО МНОГО ВРЕМЕНИ] ---")
                
                last_ts = curr_ts
                
                # Форматирование
                try:
                    dt = datetime.fromtimestamp(curr_ts)
                    date_pretty = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_pretty = "Unknown"

                role = msg.get('role', 'Unknown')
                text = msg.get('content', '').replace('\n', ' ')
                valid_msgs.append(f"[{date_pretty}] {role}: {text}")

            if valid_msgs:
                # Фильтруем пустые блоки (которые могли возникнуть из-за фильтра даты)
                cleaned_block = "\n".join(valid_msgs)
                if "[" in cleaned_block: # Простая проверка, что есть сообщения
                    final_output.append(cleaned_block)

        return "\n\n--- [НОВЫЙ ДИАЛОГ] ---\n\n".join(final_output)

    def find_best_sticker(self, query_text: str) -> Optional[Dict[str, str]]:
        if not query_text or len(query_text) < 2: return None
        query_vec = self.encoder.encode([query_text], convert_to_numpy=True).tolist()
        results = self.sticker_collection.query(query_embeddings=query_vec, n_results=1)
        if not results['ids'] or not results['ids'][0]: return None
        if results['distances'][0][0] > STICKER_THRESHOLD: return None
        meta = results['metadatas'][0][0]
        return {"path": meta.get('path'), "type": meta.get('type')}

if __name__ == "__main__":
    retriever = RagRetriever()
    print("\n--- TEST ---")
    ctx = retriever.search_context("Когда последний раз мы смотрели дораму?")
    if ctx:
        print(ctx)
    else:
        print("Ничего не найдено.")