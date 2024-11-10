from transformers import T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer

from datasets import Dataset


MODEL_NAME = "ai-forever/rut5-base"
DATASET_PATH = "datasets/data_ru_lang.dataset"

RESUME_FROM_CHECKPOINT = False
CHECKPOINT_PATH = 'train_checkpoints/'
LOG_PATH = 'train_logs/'
SAVE_MODEL_PATH = 'train_checkpoints/result'


# Before train: run data_json_creator.py
#               & dataset_tokenizer.py
# To create train data


def main():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    dataset = Dataset.load_from_disk(DATASET_PATH)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=128,  # Подбирайте в зависимости от памяти GPU
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_dir=LOG_PATH,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,  # Включить, если используете GPU NVIDIA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    print(result)
    trainer.save_model(SAVE_MODEL_PATH)


if __name__ == "__main__":
    main()
