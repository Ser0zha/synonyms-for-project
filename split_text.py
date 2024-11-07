import re
from typing import Generator


ENDING_MARKS = '.!?"'
PUNCTUATION_MARKS = ',()[]{}-:;' + ENDING_MARKS
MIN_LINE_LEN = 50
MIN_SENTENCE_LEN = 5
FILE_NAME = 'texts/war-n-piece.txt'
OUTPUT_NAME = 'texts/sents_war_n_piece.txt'


def split_line(line: str) -> list[str]:
    for i in PUNCTUATION_MARKS:
        line = line.replace(i, f' {i} ')

    return [i for i in re.split(r'\s+', line) if i]


def split_to_sentences(tokens: list[str]) -> Generator[list[str], None, None]:
    def starts_with_cap(s: str) -> bool:
        return s[0].isupper()

    last_i = 0

    prev = ""
    before_dot = ""

    for i in range(1, len(tokens)):
        before_dot = prev
        prev = tokens[i-1]
        curr = tokens[i]

        if prev not in ENDING_MARKS:
            continue

        if len(before_dot) <= 1: # probably is name
            continue

        if not starts_with_cap(curr):
            continue
        
        sentence = tokens[last_i:i]
        if len(sentence) >= MIN_SENTENCE_LEN:
            yield sentence

        last_i = i


def parse_sentences(text: str) -> Generator[list[str], None, None]:
    for line in text.split('\n'):
        if len(line) < MIN_LINE_LEN:
            continue

        tokens = split_line(line)
        yield from split_to_sentences(tokens)


def main():
    with open(FILE_NAME, 'r') as file:
        text = file.read()

    text = text.replace('…', '...')\
               .replace('«', '"')  \
               .replace('»', '"')  \
               .replace('–', '-')

    sentences = parse_sentences(text)
    
    with open(OUTPUT_NAME, 'w') as file:
        for i in sentences:
            file.write(f'{" ".join(i)}\n')

if __name__ == '__main__':
    main()
