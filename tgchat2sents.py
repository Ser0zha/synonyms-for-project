from sentence_parse_util import sentences
from argparse import ArgumentParser, FileType
import json


MIN_WORDS = 2


def get_sents(messages):
    for msg in messages:
        text = msg['text']
        
        if isinstance(text, list):
            text = ' '.join(i['text'] if isinstance(i, dict) else i for i in text)

        if text:
            yield from sentences(text)


def main():
    parser = ArgumentParser()
    parser.add_argument('filename', type=FileType('r'))

    args = parser.parse_args()
    filename = args.filename.name

    with open(filename, 'r', encoding='utf8') as file:
        input_json = json.load(file)

    sents = get_sents(input_json['messages'])

    with open(filename + '.sents', 'w', encoding='utf8') as file:
        for s in sents:
            if len([i for i in s if len(i) >= 2]) < MIN_WORDS:
                continue

            file.write(f'{" ".join(s)}\n')


if __name__ == '__main__':
    main()
