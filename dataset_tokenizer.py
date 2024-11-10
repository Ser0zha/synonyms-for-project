from datasets import Dataset
from transformers import T5Tokenizer
import json


MODEL_NAME = "ai-forever/rut5-base"
DATASET_PATH = "datasets/data_ru_lang.json"
OUTPUT_PATH = "datasets/data_ru_lang.dataset"


def load_dataset() -> Dataset:
    with open(DATASET_PATH, 'r') as file:
        data = json.load(file)
    
    return Dataset.from_list([
        {
            'input': 'WORD: "{i["word"]}", SENTENCE: {i["text"].replace("WORD", "<extra_id_0>")}', 
            'target': i['answer']
        }
        for i in data
    ])


def tokenized(data: Dataset, tokenizer) -> Dataset:
    def f(batch):
        inputs = tokenizer(batch['input'], max_length=512, truncation=True, padding=True, return_tensors='pt')
        labels = tokenizer(batch['target'], max_length=10, truncation=True, padding=True, return_tensors='pt')
        return {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': labels.input_ids
        }

    return data.map(f, batched=True)


def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    dataset = load_dataset()
    dataset = tokenized(dataset, tokenizer)
    dataset.save_to_disk(OUTPUT_PATH)


if __name__ == '__main__':
    main()
