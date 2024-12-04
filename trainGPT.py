from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import config_GPT as cfg
from datasets import Dataset

# Конфигурация
DATASET_PATH = "datasets/data_ru_lang.dataset"  # Путь к вашему текстовому файлу
OUTPUT_DIR = "gpt_model"  # Директория для сохранения модели
RESUME_FROM_CHECKPOINT = False


def tokenize_function(examples, tokenizer):
    """
    Токенизация данных.
    """
    return tokenizer(examples["text"], truncation=True, max_length=cfg.INPUT_SIZE, padding="max_length")


def main():
    # Загружаем токенайзер и модель
    tokenizer = AutoTokenizer.from_pretrained(cfg.NAME)
    model = AutoModelForCausalLM.from_pretrained(cfg.NAME)

    # Загружаем и обрабатываем данные
    dataset = Dataset.load_from_disk(DATASET_PATH)
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

    # Data collator для случайной маскировки
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Настройки обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_steps=250,  # Шаги оценки
        logging_steps=250,  # Шаги логирования
        save_steps=1000,  # Шаги сохранения модели
        save_total_limit=2,  # Максимум сохранённых моделей
        per_device_train_batch_size=52,  # Размер батча на устройство
        per_device_eval_batch_size=52,  # Размер батча для оценки
        num_train_epochs=13,  # Количество эпох
        # weight_decay=0.01,  # Весовое затухание
        # learning_rate=5e-5,  # Скорость обучения
        # warmup_steps=500,  # Тёплый старт
        logging_dir="./logs",  # Директория для TensorBoard
        fp16=True,  # Использование смешанной точности
        push_to_hub=False  # Не загружаем на Hub
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Обучение модели
    result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    print(result)
    # Сохраняем финальную модель
    trainer.save_model(OUTPUT_DIR)
    print(f"Модель сохранена в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
