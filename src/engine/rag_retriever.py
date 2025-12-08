import os
import json
import math
import re
import torch
import chromadb
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
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
STICKER_SEARCH_K = 8
# НОВОЕ: Максимальный разрыв времени внутри одного блока контекста (в секундах)
# 43200 секунд = 12 часов. Если разрыв больше, контекст обрывается.
MAX_TIME_GAP = 43200 

MAX_CONTEXT_BLOCKS = 5
TIME_DECAY_LAMBDA = 0.001
MAX_MESSAGES_PER_BLOCK = 60
MAX_TOTAL_MESSAGES = 250
MAX_TOTAL_CHARS = 16000

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
        self._query_cache = OrderedDict()
        self._max_query_cache_size = 256

    def _format_query(self, text: str) -> str:
        if not text:
            return "query:"
        stripped = text.lstrip()
        if stripped.lower().startswith("query:"):
            return text
        return f"query: {text}"

    def _format_passage(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.lstrip()
        if stripped.lower().startswith("passage:"):
            return text
        return f"passage: {text}"

    def _get_query_embedding(self, query: str) -> np.ndarray:
        formatted_query = self._format_query(query)
        cache_key = formatted_query
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return cached

        embedding = self.encoder.encode(
            [formatted_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        self._query_cache[cache_key] = embedding
        if len(self._query_cache) > self._max_query_cache_size:
            self._query_cache.popitem(last=False)
        return embedding

    def _extract_relative_bounds(self, query: str) -> Optional[Tuple[float, float]]:
        text = query.lower()
        now = datetime.now()

        day = None
        if "сегодня" in text:
            day = now
        elif "вчера" in text:
            day = now - timedelta(days=1)
        elif "позавчера" in text:
            day = now - timedelta(days=2)

        if day is not None:
            start_of_day = day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            end_of_day = day.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
            return (start_of_day, end_of_day)

        m = re.search(r"последн(?:ие|их|ю)\s+(\d+)\s+дн", text)
        if m:
            try:
                n = int(m.group(1))
            except ValueError:
                n = 0
            if n > 0:
                start_ts = (now - timedelta(days=n)).timestamp()
                end_ts = now.timestamp()
                return (start_ts, end_ts)

        if "последнюю неделю" in text or "за неделю" in text or "за последнюю неделю" in text:
            start_ts = (now - timedelta(days=7)).timestamp()
            end_ts = now.timestamp()
            return (start_ts, end_ts)

        if "последний месяц" in text or "за месяц" in text or "за последний месяц" in text:
            start_ts = (now - timedelta(days=30)).timestamp()
            end_ts = now.timestamp()
            return (start_ts, end_ts)

        if "последний год" in text or "за год" in text or "за последний год" in text:
            start_ts = (now - timedelta(days=365)).timestamp()
            end_ts = now.timestamp()
            return (start_ts, end_ts)

        return None

    def _extract_date_bounds(self, query: str) -> Optional[Tuple[float, float]]:
        """Возвращает start_ts и end_ts если найдена дата"""
        dates = search_dates(query, languages=['ru', 'en'])
        if not dates: return None
        
        date_objs = [d[1] for d in dates]
        date_objs.sort()
        start_obj = date_objs[0]
        end_obj = date_objs[-1]
        start_of_day = start_obj.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end_of_day = end_obj.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        
        if start_obj.date() == end_obj.date():
            print(f"📅 Режим даты: {start_obj.date()}")
        else:
            print(f"📅 Режим диапазона дат: {start_obj.date()} → {end_obj.date()}")
        return (start_of_day, end_of_day)

    def search_context(self, query: str) -> str:
        if not query: return ""

        # 1. Проверка даты
        date_bounds = self._extract_relative_bounds(query)
        if not date_bounds:
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

        # 2. Поиск векторов с профессиональным переранжированием
        query_embedding = self._get_query_embedding(query)
        query_vec = [query_embedding.tolist()]
        
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
        candidates = []
        score_by_index: Dict[int, float] = {}

        metadatas = results.get('metadatas')
        if metadatas and metadatas[0]:
            distances_raw = results.get('distances')
            if distances_raw and distances_raw[0]:
                distances = distances_raw[0]
            else:
                distances = [None] * len(metadatas[0])

            now_ts = datetime.now().timestamp()

            for meta, distance in zip(metadatas[0], distances):
                idx = meta.get('global_index')
                if idx is None:
                    continue
                hit_indices.add(idx)

                msg = self.full_history[idx]
                timestamp = float(msg.get('timestamp', 0)) or 0.0
                text = msg.get('content', '')

                candidates.append({
                    'index': idx,
                    'timestamp': timestamp,
                    'text': text,
                    'distance': float(distance) if distance is not None else None,
                })

            if candidates:
                texts = [self._format_passage(c['text'] or '') for c in candidates]
                doc_embeddings = self.encoder.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                sims = np.dot(doc_embeddings, query_embedding)

                for c, sim in zip(candidates, sims):
                    idx = c['index']
                    timestamp = c['timestamp']
                    if timestamp > 0:
                        age_days = max(0.0, (now_ts - timestamp) / 86400.0)
                    else:
                        age_days = 0.0

                    if TIME_DECAY_LAMBDA > 0.0:
                        decay = math.exp(-TIME_DECAY_LAMBDA * age_days)
                    else:
                        decay = 1.0

                    final_score = float(sim) * decay
                    prev_score = score_by_index.get(idx)
                    if prev_score is None or final_score > prev_score:
                        score_by_index[idx] = final_score

        if not hit_indices:
            return ""

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

        if merged_blocks and score_by_index:
            scored_blocks = []
            for start, end in merged_blocks:
                safe_end = min(end + 1, self.history_len)
                block_indices = range(start, safe_end)
                block_scores = [score_by_index[i] for i in block_indices if i in score_by_index]
                if block_scores:
                    best_score = max(block_scores)
                else:
                    best_score = 0.0

                last_ts = 0.0
                if safe_end - 1 >= 0:
                    last_msg = self.full_history[safe_end - 1]
                    last_ts_val = last_msg.get('timestamp')
                    if last_ts_val is not None:
                        last_ts = float(last_ts_val)

                scored_blocks.append((best_score, last_ts, start, end))

            scored_blocks.sort(key=lambda x: (-x[0], -x[1]))

            if MAX_CONTEXT_BLOCKS and MAX_CONTEXT_BLOCKS > 0:
                scored_blocks = scored_blocks[:MAX_CONTEXT_BLOCKS]

            merged_blocks = [(b[2], b[3]) for b in scored_blocks]

        # 4. Формирование с УМНОЙ ФИЛЬТРАЦИЕЙ
        final_output = []
        total_messages = 0
        total_chars = 0
        
        for start, end in merged_blocks:
            if MAX_TOTAL_MESSAGES and total_messages >= MAX_TOTAL_MESSAGES:
                break
            if MAX_TOTAL_CHARS and total_chars >= MAX_TOTAL_CHARS:
                break

            # Срезаем блок
            safe_end = min(end + 1, self.history_len)

            if MAX_MESSAGES_PER_BLOCK and MAX_MESSAGES_PER_BLOCK > 0:
                block_hit_indices = [i for i in sorted_hits if start <= i < safe_end]
                if block_hit_indices and score_by_index:
                    anchor_idx = max(
                        block_hit_indices,
                        key=lambda i: score_by_index.get(i, 0.0),
                    )
                elif block_hit_indices:
                    anchor_idx = block_hit_indices[len(block_hit_indices) // 2]
                else:
                    anchor_idx = (start + safe_end - 1) // 2

                half = MAX_MESSAGES_PER_BLOCK // 2
                local_start = max(start, anchor_idx - half)
                local_end = min(safe_end, anchor_idx + half + 1)
            else:
                local_start = start
                local_end = safe_end

            chunk_indices = range(local_start, local_end)
            
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
                        sep = "--- [ПРОШЛО МНОГО ВРЕМЕНИ] ---"
                        if (not MAX_TOTAL_MESSAGES or total_messages < MAX_TOTAL_MESSAGES) and (
                            not MAX_TOTAL_CHARS or (total_chars + len(sep) + 1) <= MAX_TOTAL_CHARS
                        ):
                            valid_msgs.append(sep)
                            total_messages += 1
                            total_chars += len(sep) + 1
                        else:
                            break
                
                last_ts = curr_ts
                
                # Форматирование
                try:
                    dt = datetime.fromtimestamp(curr_ts)
                    date_pretty = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_pretty = "Unknown"

                role = msg.get('role', 'Unknown')
                text = msg.get('content', '').replace('\n', ' ')
                line = f"[{date_pretty}] {role}: {text}"

                if MAX_TOTAL_MESSAGES and total_messages >= MAX_TOTAL_MESSAGES:
                    break
                if MAX_TOTAL_CHARS and (total_chars + len(line) + 1) > MAX_TOTAL_CHARS:
                    break

                valid_msgs.append(line)
                total_messages += 1
                total_chars += len(line) + 1

            if valid_msgs:
                # Фильтруем пустые блоки (которые могли возникнуть из-за фильтра даты)
                cleaned_block = "\n".join(valid_msgs)
                if "[" in cleaned_block: # Простая проверка, что есть сообщения
                    final_output.append(cleaned_block)

        return "\n\n--- [НОВЫЙ ДИАЛОГ] ---\n\n".join(final_output)

    def find_best_sticker(self, query_text: str) -> Optional[Dict[str, str]]:
        if not query_text or len(query_text) < 2:
            return None

        query_embedding = self._get_query_embedding(query_text)
        query_vec = [query_embedding.tolist()]

        try:
            results = self.sticker_collection.query(
                query_embeddings=query_vec,
                n_results=STICKER_SEARCH_K,
            )
        except Exception as e:
            print(f"Ошибка поиска стикеров: {e}")
            return None

        ids_list = results.get('ids') or []
        if not ids_list or not ids_list[0]:
            return None

        distances_list = results.get('distances') or [[]]
        metadatas_list = results.get('metadatas') or [[]]

        best_meta = None
        best_distance = None

        for meta, distance in zip(metadatas_list[0], distances_list[0]):
            if meta is None or distance is None:
                continue

            d = float(distance)
            if d > STICKER_THRESHOLD:
                continue

            if best_distance is None or d < best_distance:
                best_distance = d
                best_meta = meta

        if not best_meta:
            return None

        return {"path": best_meta.get('path'), "type": best_meta.get('type')}

if __name__ == "__main__":
    retriever = RagRetriever()
    print("\n--- TEST ---")
    ctx = retriever.search_context("Когда последний раз мы смотрели дораму?")
    if ctx:
        print(ctx)
    else:
        print("Ничего не найдено.")