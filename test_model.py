from transformers import T5ForConditionalGeneration, T5Tokenizer
import model_config as cfg
from argparse import ArgumentParser


MODEL_PATH = 'train_checkpoints/result/'


def main():
    parser = ArgumentParser()
    parser.add_argument('--text', required=True)
    parser.add_argument('--word', required=True)

    args = parser.parse_args()

    if not '___' in args.text:
        print('In text must be skipped word')
        quit()

    tokenizer = T5Tokenizer.from_pretrained(cfg.NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    prompt = cfg.prompt(args.text, args.word)

    tokenized_prompt = tokenizer(prompt, max_length=cfg.INPUT_SIZE, truncation=True, padding='max_length', return_tensors='pt')
    outputs = model.generate(**tokenized_prompt, max_length=cfg.OUTPUT_SIZE)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
