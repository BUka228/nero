import os
import sys
import json
import shutil
import subprocess
import random
from pathlib import Path
from dotenv import load_dotenv

# --- ЗАГРУЗКА КОНФИГУРАЦИИ ---
load_dotenv()

# Пути
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
MODELS_DIR = Path("models")
ADAPTERS_DIR = MODELS_DIR / "adapters"
BASE_MODEL_ID = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen3-4B-MLX-4bit")

# Настройки LoRA
TRAIN_BATCH_SIZE = int(os.getenv("LORA_BATCH_SIZE", 4)) 
TRAIN_ITERS = int(os.getenv("LORA_ITERS", 1000))
LORA_RANK = int(os.getenv("LORA_RANK", 16))

class LoraTrainer:
    def __init__(self):
        self.user_1_id = os.getenv("USER_1_ID")
        self.user_1_name = os.getenv("USER_1_NAME", "User1")
        self.user_2_id = os.getenv("USER_2_ID")
        self.user_2_name = os.getenv("USER_2_NAME", "User2")

    def _prepare_data_for_mlx(self, source_file: Path, target_dir: Path):
        """Готовит train.jsonl и valid.jsonl"""
        print(f"✂️ Подготовка данных из {source_file.name}...")
        
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Перемешиваем
        random.shuffle(lines)
        
        # Сплит 95/5
        split_idx = int(len(lines) * 0.95)
        train_data = lines[:split_idx]
        valid_data = lines[split_idx:]
        
        if len(valid_data) == 0 and len(train_data) > 1:
            valid_data = [train_data.pop()]

        with open(target_dir / "train.jsonl", 'w', encoding='utf-8') as f:
            f.writelines(train_data)
        
        with open(target_dir / "valid.jsonl", 'w', encoding='utf-8') as f:
            f.writelines(valid_data)
            
        print(f"📊 Train: {len(train_data)} строк | Valid: {len(valid_data)} строк")
        return target_dir

    def _create_lora_config(self, config_path: Path):
        """
        Создает YAML конфиг для LoRA.
        Используем плоскую структуру (Flat Structure), чтобы избежать KeyError: 'rank'.
        """
        config_content = f"""
# LoRA Configuration (Flat Structure for modern mlx-lm)
rank: {LORA_RANK}
alpha: {LORA_RANK * 2}
dropout: 0.05
keys: 
  - "self_attn.q_proj"
  - "self_attn.v_proj"
  - "self_attn.k_proj"
  - "self_attn.o_proj"
  - "mlp.gate_proj"
  - "mlp.down_proj"
  - "mlp.up_proj"
"""
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content.strip())
        print(f"⚙️ Создан конфиг LoRA: {config_path}")

    def run_training(self, user_name: str):
        """Запуск процесса обучения через CLI MLX"""
        
        source_jsonl = PROCESSED_DIR / f"train_{user_name}.jsonl"
        
        if not source_jsonl.exists():
            print(f"❌ Файл данных не найден: {source_jsonl}")
            return

        # 1. Готовим временную папку
        temp_data_path = PROCESSED_DIR / f"temp_mlx_{user_name}"
        self._prepare_data_for_mlx(source_jsonl, temp_data_path)

        # 2. Создаем конфиг файл для LoRA
        config_path = temp_data_path / "lora_config.yaml"
        self._create_lora_config(config_path)

        # 3. Путь адаптеров
        adapter_output_path = ADAPTERS_DIR / user_name
        
        print(f"\n🚀 Запуск обучения LoRA для пользователя: {user_name}")
        print(f"🤖 Базовая модель: {BASE_MODEL_ID}")
        print("-" * 50)

        # Команда запуска с флагом -c (config)
        command = [
            "mlx_lm.lora",
            "--model", BASE_MODEL_ID,
            "--train",
            "--data", str(temp_data_path),
            "--adapter-path", str(adapter_output_path),
            "--batch-size", str(TRAIN_BATCH_SIZE),
            "--iters", str(TRAIN_ITERS),
            "--save-every", "100",
            "--config", str(config_path),
            "--grad-checkpoint",
            "--seed", "42"
        ]

        try:
            subprocess.run(command, check=True)
            print("-" * 50)
            print(f"✅ Обучение завершено! Адаптеры: {adapter_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Ошибка (код {e.returncode}).")
        except FileNotFoundError:
             print("\n❌ Команда 'mlx_lm.lora' не найдена.")

    def interactive_menu(self):
        print("\n--- LoRA Training Studio (MLX) ---")
        print(f"1. Обучить модель быть: {self.user_1_name}")
        print(f"2. Обучить модель быть: {self.user_2_name}")
        print("0. Выход")
        
        choice = input("\nВыберите вариант (1/2): ").strip()
        
        if choice == "1":
            self.run_training(self.user_1_name)
        elif choice == "2":
            self.run_training(self.user_2_name)
        elif choice == "0":
            sys.exit(0)
        else:
            print("Неверный выбор.")

if __name__ == "__main__":
    trainer = LoraTrainer()
    trainer.interactive_menu()