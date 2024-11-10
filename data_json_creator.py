import json
import os
from random import choice
from typing import Generic, TypeVar

import pymorphy2

PATH_RD_WORDS = 'additional_word/russian.txt'
PATH_JSON = 'datasets/data_ru_lang.json'
PATH_TEXTS = './texts/'
MIN_WORD_LEN = 3

RD_INFLECTION_ATTEMPTS = 5
RD_FORM_VARIANTS = 1
RD_WORD_VARIANTS = 0
RD_SENTENCE_VARIANTS = 5

morph = pymorphy2.MorphAnalyzer()


class IdFactory:
    def __init__(self) -> None:
        self.next_id = 0

    def next(self):
        self.next_id += 1
        return self.next_id


_T = TypeVar('_T')


class RandomChoicer(Generic[_T]):
    def __init__(self, choices: list[_T] | tuple[_T]) -> None:
        self.choices = choices

    def get(self) -> _T:
        return choice(self.choices)


def read_lines(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        return file.readlines()


def write_json(data):
    with open(PATH_JSON, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def random_inflection(parsed_normal: pymorphy2.analyzer.Parse) -> set[str] | None:
    inflect_with = set()

    if 'NOUN' in parsed_normal.tag:
        inflect_with.add(choice((
            'nomn', 'gent', 'datv', 'accs', 'ablt', 'loct'
        )))
        inflect_with.add(choice((
            'sing', 'plur'
        )))
        return inflect_with

    if 'VERB' in parsed_normal.tag:
        inflect_with.add(choice((
            'indc', 'impr'
        )))
        # inflect_with.add(choice((
        #    'perf', 'impf'
        # )))
        inflect_with.add(choice((
            'pres', 'past', 'futr'
        )))

        if 'past' in inflect_with:
            if 'indc' in inflect_with:
                inflect_with.add(choice((
                    'masc', 'femn', 'neut'
                )))
        else:
            inflect_with.add(choice((
                '1per', '2per', '3per'
            )))
        return inflect_with

    if 'ADJF' in parsed_normal.tag:
        inflect_with.add(choice((
            'sing', 'plur'
        )))
        inflect_with.add(choice((
            'nomn', 'gent', 'datv', 'accs', 'ablt', 'loct'
        )))

        if 'sing' in inflect_with:
            inflect_with.add(choice((
                'masc', 'femn', 'neut'
            )))
        return inflect_with

    if 'ADJS' in parsed_normal.tag:
        inflect_with.add(choice((
            'sing', 'plur'
        )))

        if 'sing' in inflect_with:
            inflect_with.add(choice((
                'masc', 'femn', 'neut'
            )))
        return inflect_with


def random_word_form(word: str) -> str:
    parsed = morph.parse(word)[0].normalized  # type: ignore

    for _ in range(RD_INFLECTION_ATTEMPTS):  # attempts to inflect
        inflection = random_inflection(parsed)  # type: ignore
        if not inflection:
            return word

        inflected = parsed.inflect(inflection)  # type: ignore
        if inflected:
            return inflected[0]

    print(f'Cannot inflect in {RD_INFLECTION_ATTEMPTS} attempts', word)
    return word


def random_word_choice(sentence: str) -> tuple[str, str] | None:
    tokens = sentence.split()
    idx_to_select = [i for i, v in enumerate(tokens) if
                     len(v) >= MIN_WORD_LEN and v.isalpha() and v.islower()]  # really need is lower?

    if not idx_to_select:
        return None

    index = choice(idx_to_select)
    word = tokens[index]
    tokens[index] = 'WORD'

    return " ".join(tokens), word


def generate_word_variants(id_factory: IdFactory, random_words: RandomChoicer[str], data: list, sentence: str,
                           word: str):
    data.append({
        'id': id_factory.next(),
        'text': sentence,
        'word': word,
        'correct': True
    })

    for _ in range(RD_FORM_VARIANTS):
        if (rd_form := random_word_form(word)) != word:
            data.append({
                'id': id_factory.next(),
                'text': sentence,
                'word': rd_form.strip(),
                'correct': False
            })

    for _ in range(RD_WORD_VARIANTS):
        if (rd_word := random_words.get()) != word:
            data.append({
                'id': id_factory.next(),
                'text': sentence,
                'word': rd_word.strip(),
                'correct': False
            })


def generate_sentence_variants(sentence: str) -> set[tuple[str, str]]:
    ret = set()

    for _ in range(RD_SENTENCE_VARIANTS):
        if not (res := random_word_choice(sentence)):
            return ret
        ret.add(res)

    return ret


def generate_data() -> list[dict[str, int]]:
    id_factory = IdFactory()
    random_words = RandomChoicer(read_lines(PATH_RD_WORDS))
    data = []

    for path in os.listdir(PATH_TEXTS):
        if not path.endswith(".sents"):
            continue

        for sentence in read_lines(f'{PATH_TEXTS}/{path}'):
            for new_text, word in generate_sentence_variants(sentence):
                generate_word_variants(id_factory, random_words, data, new_text, word)

    return data


def main():
    print('Generating data...')
    data = generate_data()
    print('Saving to json...')
    write_json(data)


if __name__ == '__main__':
    main()
