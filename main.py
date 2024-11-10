from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer

from data_file import open_json
from datasets import Dataset

model_name = "ai-forever/rut5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def preprocess_data(data):
    text = data["text"].repalce("WORD", "<extra_id_0>")
    target = data["word"]
    input_text = f"Заполни пропуск: {text}"
    target_text = f"{target}" if data["correct"] == 1.0 else ""  # Пустая строка для неверных вариантов

    dict_data = {
        "input_text": input_text,
        "target_text": target_text
    }
    return dict_data


def tokenize_of_data(batch):
    inputs = tokenizer(batch["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(batch["target_text"], max_length=10, truncation=True, padding="max_length")
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = labels
    return batch


def main():
    data_set = Dataset.from_file(open_json())
    data_set = data_set.map(preprocess_data)

    tokenized_dataset = data_set.map(tokenize_of_data, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Подбирайте в зависимости от памяти GPU
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,  # Включить, если используете GPU NVIDIA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Можно использовать отдельный тестовый датасет
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
