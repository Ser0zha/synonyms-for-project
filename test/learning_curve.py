import json

import matplotlib.pyplot as plt
import torch
from evaluate import load as load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer

import model_config as cfg

CHECKPOINT_PATH = "train_checkpoints/checkpoint-11203"
TEST_CASE_PATH = "datasets/data_ru_lang.json"


def open_json():
    with open(TEST_CASE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    def evaluate_model_pass():
        all_prediction, all_label = [], []
        for prompt in texts:
            tokenized_prompt = tokenizer(
                prompt,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=cfg.INPUT_SIZE
            )

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=tokenized_prompt['input_ids'],
                    attention_mask=tokenized_prompt['attention_mask'],
                    max_length=cfg.OUTPUT_SIZE
                )

                predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)

            all_predictions.append(predictions)
            all_labels.append(prompt[1])
        return all_prediction, all_label

    def plotting_a_graph():
        # Построение графика
        plt.plot(checkpoints, accuracies, marker='o')
        plt.xlabel("Checkpoint")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve based on Checkpoints")
        plt.xticks(rotation=15)
        plt.grid()
        plt.show()

    # 1
    tokenizer = T5Tokenizer.from_pretrained(cfg.NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH, from_safetensors=True)

    # 2
    data_examples = open_json()
    texts = [[item["text"], item["answer"]] for item in data_examples]

    # 3
    metric = load_metric("accuracy")  # Или подходящую метрику
    model.eval()
    all_predictions, all_labels = evaluate_model_pass()

    # Вычислим метрику
    eval_results = metric.compute(predictions=all_predictions, references=all_labels)
    print(f"Accuracy on validation set: {eval_results['accuracy']}")

    # 4
    checkpoints = ["train_checkpoints/checkpoint-11000",
                   "train_checkpoints/checkpoint-11203",
                   "train_checkpoints/result"]

    accuracies = []

    for path in checkpoints:
        model = T5ForConditionalGeneration.from_pretrained(path, from_safetensors=True)
        model.eval()
        checkpoint_predictions, _ = evaluate_model_pass()
        eval_results = metric.compute(predictions=checkpoint_predictions, references=all_labels)
        accuracies.append(eval_results["accuracy"])

    plotting_a_graph()


if __name__ == "__main__":
    main()
