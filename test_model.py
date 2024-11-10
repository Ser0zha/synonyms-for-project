from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "ai-forever/rut5-base"
MODEL_PATH = 'train_checkpoints/checkpoint-1000/'


def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    prompt = 'IS IT CORRECT TO PUT WORD "писателей" INTO SENTENCE: Не бывает на свете чисто природоведческих <extra_id_0> , которые писали бы исключительно о природе .'

    tokenized_prompt = tokenizer(prompt, max_length=512, truncation=True, padding=True, return_tensors='pt')
    outputs = model.generate(**tokenized_prompt, max_length=10)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
