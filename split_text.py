import re
import sys
from typing import Generator


ENDING_MARKS = '.!?"'
PUNCTUATION_MARKS = ',()[]{}-:;@#$&*' + ENDING_MARKS
REPLACE = {
    '…': '...',
    '«': '"',
    '»': '"',
    '–': '-'
}

MIN_LINE_LEN = 20
MIN_SENTENCE_WORDS = 3


def split_line(line: str) -> list[str]:
    for i in PUNCTUATION_MARKS:
        line = line.replace(i, f' {i} ')

    return [i for i in re.split(r'\s+', line) if i]


def split_to_sentences(tokens: list[str]) -> Generator[list[str], None, None]:
    def starts_with_cap(s: str) -> bool:
        return s[0].isupper()

    last_i = 0
    i = 0

    prev = ""
    prev2 = ""
    before_dot = ""

    for i in range(1, len(tokens)):
        prev2 = prev
        prev = tokens[i-1]
        curr = tokens[i]

        if prev not in ENDING_MARKS:
            continue

        if prev2 not in PUNCTUATION_MARKS:
            before_dot = prev2

        if len(before_dot) <= 1: # probably is name
            continue

        if not starts_with_cap(curr):
            continue
        
        yield tokens[last_i:i]

        last_i = i

    if last_i != i and tokens[i] in ENDING_MARKS:
        yield tokens[last_i:]


def parse_sentences(text: str) -> Generator[list[str], None, None]:
    for line in text.split('\n'):
        if len(line) < MIN_LINE_LEN:
            continue

        tokens = split_line(line)
        yield from split_to_sentences(tokens)


def main():
    filename = sys.argv[1]

    with open(filename, 'r') as file:
        text = file.read()

    for i in REPLACE:
        text = text.replace(i, REPLACE[i])

    sentences = parse_sentences(text)
    
    with open(filename + '.sents', 'w') as file:
        for i in sentences:
            if len([j for j in i if len(j) >= 2]) < MIN_SENTENCE_WORDS:
                continue

            file.write(f'{" ".join(i)}\n')


if __name__ == '__main__':
    main()
