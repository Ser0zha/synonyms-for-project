import random
from typing import Any

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

PATH = "additional_word/words.txt"


def wa_gen() -> str:
    with open(PATH, "r", encoding="utf-8") as file:
        list_of_file = [word.strip() for word in file]
    wrong_word = random.choice(list_of_file)
    return wrong_word


def lemmatization(word: str) -> Any:
    lemma = morph.parse(word)[0].normal_form
    if lemma == word:
        return wa_gen()
    else:
        return lemma
