from datasets import Dataset
from transformers import T5Tokenizer
import json
import model_config as cfg
from argparse import ArgumentParser, FileType


def load_dataset(path: str) -> Dataset:
    with open(path, 'r', encoding='utf8') as file:
        data = json.load(file)
    
    return Dataset.from_list([
        {
            'input': cfg.prompt(i['text'], i['word']),
            'target': 'true' if i['answer'] == i['word'] else 'false'
        }
        for i in data
    ])


def tokenized(data: Dataset, tokenizer) -> Dataset:
    def f(batch):
        inputs = tokenizer(batch['input'], max_length=cfg.INPUT_SIZE, truncation=True, padding='max_length', return_tensors='pt')
        labels = tokenizer(batch['target'], max_length=cfg.OUTPUT_SIZE, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': inputs.input_ids,
            'labels': labels.input_ids
        }

    return data.map(f, batched=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=FileType('r'))
    parser.add_argument('output')

    args = parser.parse_args()

    input_filename = args.input.name
    output_path = args.output

    tokenizer = T5Tokenizer.from_pretrained(cfg.NAME, legacy=False)

    dataset = load_dataset(input_filename)
    dataset = tokenized(dataset, tokenizer)
    dataset.save_to_disk(output_path)


if __name__ == '__main__':
    main()
