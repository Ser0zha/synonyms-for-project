from sentence_parse_util import sentences
from typing import Generator
from argparse import ArgumentParser, FileType


REPLACE = {
    '…': '...',
    '«': '"',
    '»': '"',
    '–': '-'
}
MIN_LINE_LEN = 20
MIN_SENTENCE_WORDS = 3


def parse_sentences(text: str) -> Generator[list[str], None, None]:
    for line in text.split('\n'):
        if len(line) < MIN_LINE_LEN:
            continue

        yield from sentences(line)


def main():
    parser = ArgumentParser()
    parser.add_argument('filename', type=FileType('r'))

    args = parser.parse_args()

    with open(args.filename.name, 'r') as file:
        text = file.read()

    for i in REPLACE:
        text = text.replace(i, REPLACE[i])

    sentences = parse_sentences(text)
    
    with open(args.filename.name + '.sents', 'w') as file:
        for i in sentences:
            if len([j for j in i if len(j) >= 2]) < MIN_SENTENCE_WORDS:
                continue

            file.write(f'{" ".join(i)}\n')


if __name__ == '__main__':
    main()
