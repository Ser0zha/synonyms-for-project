import re

import pymorphy2
from ruwordnet import RuWordNet

import data_file

# константы
data_sent = data_file.list_of_sentence
data_answ = data_file.list_of_answer
length = len(data_sent)

# Инициализация RuWordNet и Pymorphy2
ruwordnet = RuWordNet()
morph = pymorphy2.MorphAnalyzer()


def clean_parentheses(tmp: str) -> list[str]:
    # убираем все слова - пояснения т.е., те слова, что находятся в скобках
    removing_parentheses = re.sub(r'\(.*?\)', '', tmp)

    print("\t\t" + removing_parentheses)
    removing_parentheses = removing_parentheses.strip()
    print("\t" + removing_parentheses)

    d = [removing_parentheses.lower()]

    for word in removing_parentheses.split():
        d.append(word.lower())

    return d


def additional_processing(lst: set[str]) -> list[str]:
    """Функция для доп обработка синонимов, убрать лишние скобки и т.д."""
    processed = []
    print(lst)
    for words in lst:
        if len(words.split()) > 1:
            processed.extend(clean_parentheses(words))
        else:
            processed.append(words.lower())
    print(processed)
    return list(set(processed))


def removing_unnecessary(lst: set[str]) -> list[str]:
    ret = list(lst)
    for i, val in enumerate(ret):
        ret[i] = ret[i].lower()
        sp = val.split("(")
        if len(sp) > 1:
            ret[i] = sp[0].strip().lower()
    return ret


def get_synonyms(word: str) -> list[str]:
    """Функция для поиска синонимов слова через WordNet"""
    synonyms = set()

    # Приводим слово к начальной форме (лемматизация)
    lemma = morph.parse(word)[0].normal_form  # type: ignore

    # Ищем в RuWordNet синонимы для данного слова
    synsets = ruwordnet.get_synsets(lemma)

    for synset in synsets:
        synonyms.add(synset.title)  # Добавляем синонимы
    print(synonyms)
    return removing_unnecessary(synonyms)


def getting_a_response() -> list[str]:
    """Функция для обращения к пользователю и получения ответа"""
    user_answer = []
    print("Хотите закончить попытку прежде временно? - Напиши 00 ")
    for i in range(length):

        answer = input(str(i + 1) + ") " + data_sent[i] + "\n")

        if answer == "00":
            break

        user_answer.append(answer.strip())

    return user_answer


def compare_answers(user_answers: list[str]) -> list[bool | int]:
    results = []

    for i, user in enumerate(user_answers):

        correct_word = data_answ[i].lower()

        correct_synonyms = get_synonyms(correct_word)

        if user.lower() == correct_word or user.lower() in correct_synonyms:
            results.append(True)

        else:
            results.append(i)

    return results


def announcement_of_results(res: list[bool | int]):
    print()
    for i in range(len(res)):
        print(f"---------------[{i + 1}]---------------")
        if res[i] is True:
            print("✓ - correct")
        else:
            print("⨉ - wrong" + "\n(" + data_sent[i] + ")\n")


def main():
    usr = getting_a_response()
    result = compare_answers(usr)
    announcement_of_results(result)


if __name__ == "__main__":
    main()
