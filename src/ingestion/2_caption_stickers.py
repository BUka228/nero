import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Загружаем настройки
load_dotenv()

# Пути
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/result.json")).parent
STICKERS_DIR = RAW_DATA_PATH 
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
OUTPUT_FILE = PROCESSED_DIR / "stickers_metadata.json"

# Конфигурация модели
MODEL_PATH = "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"

class StickerCaptioner:
    def __init__(self):
        print(f"🚀 Загрузка Vision-модели: {MODEL_PATH}...")
        self.model, self.processor = load(MODEL_PATH)
        self.config = load_config(MODEL_PATH)
        print("✅ Модель загружена")

        self.system_prompt = "Describe the emotion, gesture, and visual content of this sticker concisely. Focus on the mood."
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _get_original_file(self, thumb_path: Path) -> Path:
        original_name = thumb_path.name.replace("_thumb.jpg", "")
        return thumb_path.parent / original_name

    def generate_caption(self, image_path: str) -> str:
        """Генерирует описание для одного изображения"""
        try:
            formatted_prompt = apply_chat_template(
                self.processor, 
                self.config, 
                self.system_prompt, 
                num_images=1
            )
            
            # mlx_vlm возвращает объект GenerationResult
            output = generate(
                self.model, 
                self.processor, 
                formatted_prompt, 
                [image_path], 
                verbose=False
            )
            
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Проверяем, является ли output объектом с атрибутом text
            if hasattr(output, "text"):
                return output.text.strip()
            # Если это строка (в старых версиях)
            elif isinstance(output, str):
                return output.strip()
            # На всякий случай
            else:
                return str(output).strip()

        except Exception as e:
            print(f"\n⚠️ Ошибка при обработке {image_path}: {e}")
            return ""

    def process_stickers(self):
        existing_data = {}
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                try:
                    loaded = json.load(f)
                    existing_data = {item['path']: item for item in loaded}
                    print(f"📂 Загружено {len(existing_data)} уже обработанных стикеров.")
                except json.JSONDecodeError:
                    pass

        all_thumbs = list(STICKERS_DIR.rglob("*_thumb.jpg"))
        print(f"🔍 Найдено {len(all_thumbs)} файлов превью.")

        results = []
        for k, v in existing_data.items():
            results.append(v)

        new_tasks = []
        for thumb in all_thumbs:
            original_file = self._get_original_file(thumb)
            try:
                rel_path = str(original_file.relative_to(RAW_DATA_PATH))
            except ValueError:
                rel_path = original_file.name

            if rel_path in existing_data:
                continue
            
            new_tasks.append((thumb, rel_path, original_file))

        if not new_tasks:
            print("🎉 Все стикеры уже обработаны!")
            return

        print(f"📸 Начинаем генерацию описаний для {len(new_tasks)} новых стикеров...")

        for thumb_path, rel_path, original_path in tqdm(new_tasks):
            if not original_path.exists():
                continue

            description = self.generate_caption(str(thumb_path))
            
            file_type = "video" if original_path.suffix in ['.webm', '.mp4'] else "static"
            if original_path.suffix == '.tgs':
                file_type = "animated_tgs"

            entry = {
                "path": rel_path,
                "description": description,
                "type": file_type
            }
            results.append(entry)

            if len(results) % 10 == 0:
                self._save_json(results)

        self._save_json(results)
        print(f"✅ Готово! Метаданные сохранены в {OUTPUT_FILE}")

    def _save_json(self, data):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if not STICKERS_DIR.exists():
        print(f"❌ Папка {STICKERS_DIR} не найдена. Проверьте RAW_DATA_PATH в .env")
    else:
        captioner = StickerCaptioner()
        captioner.process_stickers()