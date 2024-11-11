import json
from random import choice
from typing import Generic, TypeVar

import pymorphy2
from argparse import ArgumentParser, FileType


MORPH = pymorphy2.MorphAnalyzer()


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


def write_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as file:
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


def random_word_form(word: str, attempts: int) -> str:
    parsed = MORPH.parse(word)[0].normalized  # type: ignore

    for _ in range(attempts):  # attempts to inflect
        inflection = random_inflection(parsed)  # type: ignore
        if not inflection:
            return word

        inflected = parsed.inflect(inflection)  # type: ignore
        if inflected:
            return inflected[0]

    print(f'Cannot inflect in {attempts} attempts', word)
    return word


def random_word_choice(sentence: str, min_word_len: int) -> tuple[str, str] | None:
    tokens = sentence.split()
    idx_to_select = [
        i 
        for i, v in enumerate(tokens)
        if len(v) >= min_word_len and v.isalpha() and v.islower()
    ]  # really need is lower?

    if not idx_to_select:
        return None

    index = choice(idx_to_select)
    word = tokens[index]
    tokens[index] = '___'

    return " ".join(tokens), word


def generate_word_variants(
    id_factory: IdFactory,
    random_words: RandomChoicer[str],
    data: list,
    sentence: str,
    word: str,
    form_variants: int,
    rd_word_variants: int,
    inflect_rd_words: int,
    inflect_attempts: int
):
    data.append({
        'id': id_factory.next(),
        'text': sentence,
        'word': word,
        'answer': word
    })

    for _ in range(form_variants):
        if (rd_form := random_word_form(word, inflect_attempts)) != word:
            data.append({
                'id': id_factory.next(),
                'text': sentence,
                'word': rd_form.strip(),
                'answer': word
            })

    for _ in range(rd_word_variants):
        rd_word = random_words.get()
        if inflect_rd_words:
            rd_word = random_word_form(rd_word, inflect_attempts)

        if rd_word != word:
            data.append({
                'id': id_factory.next(),
                'text': sentence,
                'word': rd_word.strip(),
                'answer': word
            })


def generate_sentence_variants(
    sentence: str,
    variants: int,
    min_word_len: int
) -> set[tuple[str, str]]:
    ret = set()

    for _ in range(variants):
        if not (res := random_word_choice(sentence, min_word_len)):
            return ret
        ret.add(res)

    return ret


def generate_data(
    input_paths: list[str],
    path_rd_words: str,
    sentence_variants: int,
    min_word_len: int,
    form_variants: int,
    rd_word_variants: int,
    inflect_rd_words: int,
    inflect_attempts: int
) -> list[dict[str, int]]:
    id_factory = IdFactory()
    random_words = RandomChoicer(read_lines(path_rd_words) if path_rd_words else [])
    data = []

    for path in input_paths:
        for sentence in read_lines(path):
            for new_text, word in generate_sentence_variants(sentence, sentence_variants, min_word_len):
                generate_word_variants(
                    id_factory,
                    random_words,
                    data,
                    new_text,
                    word,
                    form_variants,
                    rd_word_variants,
                    inflect_rd_words,
                    inflect_attempts
                )

    return data


def main():
    parser = ArgumentParser()
    parser.add_argument('infiles', nargs='+', help='input filenames', type=FileType('r'))
    parser.add_argument('outfile', help='output filename', type=FileType('w'))
    parser.add_argument('-r', help='file of random words to put in', type=FileType('r'))
    parser.add_argument('-m', help='change form of random words', action='store_true')
    parser.add_argument('-i', help='inflection/mutation attempts', type=int, default=5)
    parser.add_argument('-fv', help='generate N examples with forms of original word', type=int, default=2)
    parser.add_argument('-wv', help='generate N examples with forms of random word', type=int, default=2)
    parser.add_argument('-sv', help='generate N examples with different skipped word', type=int, default=5)
    parser.add_argument('-mw', help='minimal word length algorithm procedure', type=int, default=3)

    args = parser.parse_args()

    if args.wv != 0 and args.r is None:
        print('If you want use random words, specify random words file')
        quit()

    print('Generating data...')
    data = generate_data(
        [i.name for i in args.infiles],
        args.r.name,
        args.sv,
        args.mw,
        args.fv,
        args.wv,
        args.m,
        args.i
    )
    print('Saving to json...')
    write_json(data, args.outfile.name)


if __name__ == '__main__':
    main()
