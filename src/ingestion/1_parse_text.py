import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения
load_dotenv()

# Конфигурация из .env
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/result.json")
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
USER_1_ID = os.getenv("USER_1_ID")
USER_1_NAME = os.getenv("USER_1_NAME", "User1")
USER_1_PROMPT = os.getenv("USER_1_SYSTEM_PROMPT", "You are a helpful assistant.")
USER_2_ID = os.getenv("USER_2_ID")
USER_2_NAME = os.getenv("USER_2_NAME", "User2")
USER_2_PROMPT = os.getenv("USER_2_SYSTEM_PROMPT", "You are a helpful assistant.")
TIMEOUT_SECONDS = int(os.getenv("CONVERSATION_TIMEOUT", 7200)) # 2 часа

# Убедимся, что папка для вывода существует
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class TelegramParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Карта ID -> Имя для удобства
        self.id_map = {
            f"user{USER_1_ID}": USER_1_NAME,
            f"user{USER_2_ID}": USER_2_NAME
        }

    def _extract_text(self, msg: Dict[str, Any]) -> str:
        """Извлекает чистый текст из сложной структуры Telegram"""
        text_content = ""
        
        # 1. Обработка текстового поля
        if "text" in msg:
            if isinstance(msg["text"], str):
                text_content = msg["text"]
            elif isinstance(msg["text"], list):
                for entity in msg["text"]:
                    if isinstance(entity, str):
                        text_content += entity
                    elif isinstance(entity, dict) and "text" in entity:
                        text_content += entity["text"]

        # 2. Обработка медиа
        media_tag = ""
        
        # Безопасно получаем тип медиа (если ключа нет, вернет None)
        media_type = msg.get("media_type")

        if media_type == "sticker":
            emoji = msg.get("sticker_emoji", "")
            file_path = msg.get("file", "")
            media_tag = f" [STICKER: {emoji} path={file_path}]"
        
        elif media_type == "video_file":
             media_tag = " [VIDEO_MESSAGE]"
        
        # Проверяем наличие фото через .get(), чтобы избежать KeyError
        elif msg.get("photo"):
            media_tag = " [PHOTO]"

        full_text = text_content + media_tag
        return full_text.strip()

    def process_chat(self):
        """
        Основная логика:
        1. Чистит мусор.
        2. Группирует сообщения от одного автора, идущие подряд.
        3. Разбивает на сессии по времени.
        """
        print(f"🔄 Начало обработки: {self.data.get('name', 'Unknown Chat')}")
        
        cleaned_messages = []
        raw_msgs = self.data.get("messages", [])
        
        if not raw_msgs:
            print("⚠️ Ошибка: Нет сообщений в JSON")
            return

        # Временные переменные для группировки
        buffer_text = []
        last_user_id = None
        last_timestamp = 0
        last_date_str = ""
        
        for msg in raw_msgs:
            # Пропускаем системные сообщения (не имеющие from_id)
            if msg["type"] != "message" or "from_id" not in msg:
                continue

            current_user_id = msg["from_id"]
            current_timestamp = int(msg["date_unixtime"])
            text = self._extract_text(msg)
            
            # Если текста нет и нет медиа (пустое сообщение) - пропускаем
            if not text:
                continue

            # Логика группировки (Grouping Logic)
            is_same_user = (current_user_id == last_user_id)
            is_small_gap = (current_timestamp - last_timestamp) < 300 # 5 минут на склейку сообщений подряд
            
            if is_same_user and is_small_gap:
                # Добавляем в буфер текущего сообщения (склеиваем через пробел или \n)
                buffer_text.append(text)
                # Обновляем время последнего сообщения в группе
                last_timestamp = current_timestamp 
            else:
                # Если сменился юзер или прошла куча времени -> Сохраняем предыдущий блок
                if buffer_text and last_user_id:
                    cleaned_messages.append({
                        "role": self.id_map.get(last_user_id, "unknown"),
                        "user_id": last_user_id,
                        "content": "\n".join(buffer_text), # Склеиваем переносом строки
                        "timestamp": last_timestamp,
                        "date": last_date_str
                    })
                
                # Начинаем новый блок
                buffer_text = [text]
                last_user_id = current_user_id
                last_timestamp = current_timestamp
                last_date_str = msg["date"]

        # Не забываем сохранить последний кусок
        if buffer_text and last_user_id:
            cleaned_messages.append({
                "role": self.id_map.get(last_user_id, "unknown"),
                "user_id": last_user_id,
                "content": "\n".join(buffer_text),
                "timestamp": last_timestamp,
                "date": last_date_str
            })

        print(f"✅ Сгруппировано сообщений: {len(cleaned_messages)}")
        return cleaned_messages

    def export_for_rag(self, messages):
        """Сохраняет простой JSON для поиска"""
        output_path = PROCESSED_DIR / "chat_history_rag.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"💾 RAG dataset сохранен: {output_path}")

    def export_for_finetuning(self, messages):
        """
        Генерирует JSONL файлы для обучения (ChatML формат).
        Делает ДВА файла: 
        1. Обучение модели быть User1
        2. Обучение модели быть User2
        """
        
        def save_jsonl(target_id: str, system_prompt: str, filename: str):
            data_rows = []
            
            # target_id должен быть строкой, как в from_id (например "user12345")
            target_uid_str = f"user{target_id}"
            
            for i in range(len(messages) - 1):
                msg_prev = messages[i]
                msg_next = messages[i+1]
                
                # Проверка на разрыв времени (если между сообщениями > 2 часов, контекст потерян)
                if (msg_next["timestamp"] - msg_prev["timestamp"]) > TIMEOUT_SECONDS:
                    continue

                # ЛОГИКА ОБУЧЕНИЯ:
                # Если следующее сообщение написал Target, то предыдущее - это Input (User), а следующее - Output (Assistant)
                if msg_next["user_id"] == target_uid_str:
                    entry = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": msg_prev["content"]},
                            {"role": "assistant", "content": msg_next["content"]}
                        ]
                    }
                    data_rows.append(entry)
            
            out_path = PROCESSED_DIR / filename
            with open(out_path, 'w', encoding='utf-8') as f:
                for row in data_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"🎓 Training dataset ({filename}) сохранен: {len(data_rows)} примеров")

        # Генерируем для Первого пользователя
        if USER_1_ID:
            save_jsonl(USER_1_ID, USER_1_PROMPT, f"train_{USER_1_NAME}.jsonl")
        
        # Генерируем для Второго пользователя
        if USER_2_ID:
            save_jsonl(USER_2_ID, USER_2_PROMPT, f"train_{USER_2_NAME}.jsonl")

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Файл {RAW_DATA_PATH} не найден! Положите result.json в папку data/raw/")
    else:
        parser = TelegramParser(RAW_DATA_PATH)
        processed_msgs = parser.process_chat()
        
        if processed_msgs:
            parser.export_for_rag(processed_msgs)
            parser.export_for_finetuning(processed_msgs)