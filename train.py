from transformers import T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import model_config as cfg


DATASET_PATH = "datasets/data_ru_lang.dataset"

RESUME_FROM_CHECKPOINT = False
CHECKPOINT_PATH = 'train_checkpoints/'
SAVE_MODEL_PATH = 'train_checkpoints/result'


# Before train: run data_json_creator.py
#               & dataset_tokenizer.py
# To create train data


def main():
    model = T5ForConditionalGeneration.from_pretrained(cfg.NAME)

    dataset = Dataset.load_from_disk(DATASET_PATH)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=52,  # Подбирайте в зависимости от памяти GPU
        gradient_accumulation_steps=4,
        num_train_epochs=17,
        save_total_limit=2,
        save_steps=250,
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
