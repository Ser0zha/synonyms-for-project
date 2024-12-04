import json

import matplotlib.pyplot as plt
import torch
from evaluate import load as load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer

import model_config as cfg

CHECKPOINT_PATH = "train_checkpoints/checkpoint-11203"
TEST_CASE_PATH = "datasets/data_ru_for_test.json"


def open_json():
    with open(TEST_CASE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        # Собираем текст + слово и метку
        return [[f"{item['text']} {item['word']}", int(item["answer"])] for item in data]


def evaluate_model_pass(model, tokenizer, texts):
    all_prediction, all_label = [], []

    for text, label in texts:
        tokenized_prompt = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.INPUT_SIZE
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_prompt["input_ids"],
                attention_mask=tokenized_prompt["attention_mask"],
                max_length=cfg.OUTPUT_SIZE
            )

            predictions = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Преобразуем предсказания в числовой формат
            if predictions == "true":
                predictions = 1
            elif predictions == "false":
                predictions = 0

            print(f"Predicted: {predictions}, Label: {label}")

        all_prediction.append(predictions)
        all_label.append(label)

    return all_prediction, all_label


def plotting_a_graph(checkpoints, accuracies):
    plt.plot(checkpoints, accuracies, marker="o")
    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve based on Checkpoints")
    plt.xticks(rotation=15)
    plt.grid()
    plt.show()


def main():
    # 1. Загрузка модели и токенайзера
    tokenizer = T5Tokenizer.from_pretrained(cfg.NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)

    # 2. Загрузка тестовых данных
    texts = open_json()

    # 3. Оценка метрики
    metric = load_metric("accuracy")
    model.eval()

    all_predictions, all_labels = evaluate_model_pass(model, tokenizer, texts)
    eval_results = metric.compute(predictions=all_predictions, references=all_labels)
    print(f"Accuracy on validation set: {eval_results['accuracy']}")

    # 4. Тестирование на контрольных точках
    checkpoints = [
        "train_checkpoints/checkpoint-11000",
        "train_checkpoints/checkpoint-11203",
        "train_checkpoints/result"
    ]
    accuracies = []

    for path in checkpoints:
        print(f"Evaluating checkpoint: {path}")
        model = T5ForConditionalGeneration.from_pretrained(path)
        model.eval()
        checkpoint_predictions, _ = evaluate_model_pass(model, tokenizer, texts)
        eval_results = metric.compute(predictions=checkpoint_predictions, references=all_labels)
        accuracies.append(eval_results["accuracy"])
        print(accuracies)
    plotting_a_graph(checkpoints, accuracies)


if __name__ == "__main__":
    main()
