import json
import torch

import matplotlib.pyplot as plt
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def open_json():
    with open("datasets/data_ru_lang.json", "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # 1
    checkpoint_path = "train_checkpoints/checkpoint-11203"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    # 2
    data_examples = open_json()
    texts = [[item['text'], item["answer"]] for item in data_examples]

    # 3
    metric = load_metric("accuracy")  # Или подходящую метрику
    model.eval()
    all_predictions, all_labels = [], []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        true_label = text['label']
        all_predictions.append(predictions)
        all_labels.append(true_label)

    # Вычислим метрику
    eval_results = metric.compute(predictions=all_predictions, references=all_labels)
    print(f"Accuracy on validation set: {eval_results['accuracy']}")

    # 4
    checkpoints = ["train_checkpoints/checkpoint-11000",
                   "train_checkpoints/checkpoint-11203"]
    accuracies = []

    for path in checkpoints:
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        eval_results = metric.compute(predictions=all_predictions, references=all_labels)
        accuracies.append(eval_results["accuracy"])
    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve based on Checkpoints")
    plt.show()


if __name__ == "__main__":
    main()
