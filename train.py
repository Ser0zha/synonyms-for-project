import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer

import model_config as cfg
from datasets import Dataset

DATASET_PATH = "datasets/data_ru_lang.dataset"
RESUME_FROM_CHECKPOINT = False
CHECKPOINT_PATH = 'train_checkpoints/'
SAVE_MODEL_PATH = 'train_checkpoints/result'


# Before train: run data_json_creator.py
#               & dataset_tokenizer.py
# To create train data

# Функция для инициализации процесса
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Адрес главного узла (можно указать IP адрес сервера)
    os.environ['MASTER_PORT'] = '12355'  # Порт для связи между процессами
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # Устанавливаем устройство (GPU) для текущего процесса


def cleanup():
    dist.destroy_process_group()


def main():
    torch.cuda.empty_cache()  # Очистка кэша GPU

    # Установка устройства
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()

    # Инициализация распределённого процесса
    setup(rank, world_size)

    model = T5ForConditionalGeneration.from_pretrained(cfg.NAME)

    model = DDP(model.to(rank), device_ids=[rank], output_device=rank)

    # Загрузка датасета
    dataset = Dataset.load_from_disk(DATASET_PATH)

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=44,  # Подбирайте в зависимости от памяти GPU
        gradient_accumulation_steps=4,
        num_train_epochs=12,
        save_total_limit=2,
        save_steps=250,
        fp16=True,  # Включить, если используете GPU NVIDIA
        learning_rate=5e-5,  # Начальный LR
        lr_scheduler_type="linear",  # Линейное затухание
        warmup_steps=500,  # Тёплый старт для стабильного обучения
        remove_unused_columns=False,
        max_grad_norm=1.0  # Ограничение на максимальные градиенты
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    print(result)
    trainer.save_model(SAVE_MODEL_PATH)

    # Завершаем процесс
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.environ["LOCAL_RANK"] = str(0)  # Укажите текущий ранг (0 для первого процесса)
    main()
