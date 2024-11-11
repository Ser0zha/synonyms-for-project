from typing import Generator
import re
from emoji import EMOJI_DATA


EMOJI_LIST = '|'.join(i for i in EMOJI_DATA.keys() if len(i) == 1)
EMOJI_RE = re.compile(EMOJI_LIST)

ENDING_MARKS = '.!?"'
PUNCTUATION_MARKS = ',()[]{}-:;@#$&*\'|_' + ENDING_MARKS


def _split_line(line: str) -> list[str]:
    for i in PUNCTUATION_MARKS:
        line = line.replace(i, f' {i} ')

    line = EMOJI_RE.sub('', line)
    
    return [i for i in re.split(r'\s+', line) if i]


def _split_to_sentences(tokens: list[str]) -> Generator[list[str], None, None]:
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

    if last_i != i:
        yield tokens[last_i:]


def sentences(line: str) -> Generator[list[str], None, None]:
    return _split_to_sentences(_split_line(line))
