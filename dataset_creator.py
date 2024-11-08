import json
import os
import random

PATH_JSON = "data_ru_lang.json"
PATH_TEXTS = "./texts/"
MIN_WORD_LEN = 3


class IdFactory:
    def __init__(self) -> None:
        self.next_id = 0

    def next(self):
        self.next_id += 1
        return self.next_id


def read_lines(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        return file.readlines()


def write_json(data):
    with open(PATH_JSON, "w+", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def random_word_choice(sentence: str) -> tuple[str, str, bool] | None:
    tokens = sentence.split()
    idx_to_select = [i for i, v in enumerate(tokens) if len(v) >= MIN_WORD_LEN and v.isalpha() and v.islower()] # really need is lower?

    if not idx_to_select:
        return None

    index = random.choice(idx_to_select)
    word = tokens[index]
    tokens[index] = "WORD"

    return " ".join(tokens), word, True # TODO: bad answers


def generate_data() -> list:
    id_factory = IdFactory()
    data = []

    for path in os.listdir(PATH_TEXTS):
        if not path.endswith(".sents"):
            continue
        
        for sentence in read_lines(f'{PATH_TEXTS}/{path}'):
            if not (res := random_word_choice(sentence)):
                continue

            new_text, word, is_correct = res
            data.append({
                'id': id_factory.next(),
                'text': new_text,
                'word': word,
                'correct': is_correct
            })

    return data


def main():
    data = generate_data()
    write_json(data)


if __name__ == '__main__':
    main()
