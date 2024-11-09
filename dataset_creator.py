import json
import os

from additional_word.wrong_answ_generator import *

PATH_JSON = "data_ru_lang1.json"
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
    response_indicator = True

    tokens = sentence.split()
    idx_to_select = [i for i, v in enumerate(tokens) if
                     len(v) >= MIN_WORD_LEN and v.isalpha() and v.islower()]  # really need is lower?

    if not idx_to_select:
        return None

    index = random.choice(idx_to_select)
    word = tokens[index]
    tokens[index] = "WORD"

    return " ".join(tokens), word, response_indicator  # TODO: bad answers


def generate_data() -> list[dict[str, int]]:
    id_factory = IdFactory()
    data = []

    for path in os.listdir(PATH_TEXTS):
        if not path.endswith(".sents"):
            continue

        for sentence in read_lines(f'{PATH_TEXTS}/{path}'):
            if not (res := random_word_choice(sentence)):
                continue

            new_text, word, is_correct = res

            for i in range(4):

                dict_template = {
                    'id': id_factory.next(),
                    'text': new_text,
                    'word': str,
                    'correct': bool
                }

                match i:
                    case 0:
                        dict_template["word"] = word
                        dict_template["correct"] = True
                    case 1:
                        dict_template["word"] = wa_gen()
                        dict_template["correct"] = False
                    case 2:
                        dict_template["word"] = lemmatization(word)
                        dict_template["correct"] = False
                    case 3:
                        dict_template["word"] = wa_gen()
                        dict_template["correct"] = False

                data.append(dict_template)
    return data


def main():
    data = generate_data()
    write_json(data)


if __name__ == '__main__':
    main()
