import json
import os
import random

PATH_JSON = "data_ru_lang.json"
PATH_TEXTS = "./texts/"


def load_json():
    """Загружает данные из JSON-файла."""
    with open(PATH_JSON, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_json(list_of_res):
    with open(PATH_JSON, "w+", encoding="utf-8") as file:
        json.dump(list_of_res, file, ensure_ascii=False, indent=4)


def get_next_task_id(tasks):
    """Возвращает следующий доступный taskID."""
    if not tasks:
        return 1
    return max(task.get("taskID", 0) for task in tasks) + 1


def add_task(text, answer):
    """
    :param text: Текст задачи с пропуском.
    :param answer: Правильный ответ на задачу.
    """
    data = load_json()
    tasks = data["zadanie1"]

    new_task_id = get_next_task_id(tasks)

    new_task = {
        "taskID": new_task_id,
        "text": text,
        "answer": answer
    }

    tasks.append(new_task)

    write_json(data)


def random_word_choice(sentences: str) -> list[str]:
    def check_alpha(_: str) -> bool:
        return _.isalpha()

    listik = sentences.split()
    list_of_numbers = []

    i = 0
    for sent in sentences.split():
        length = len(sent)
        if length > 3 and check_alpha(sent) and sent.islower():
            list_of_numbers.append(i)
        i += 1

    if len(list_of_numbers) == 0:
        return ["err", "-"]

    index = random.choice(list_of_numbers)
    word = listik[index]
    listik[index] = "WORD"

    return [" ".join(listik), word]


def read_text_dir():
    def open_text() -> list[str]:
        with open(PATH_TEXTS + path, "r", encoding="utf-8") as file:
            db = [i.strip() for i in file.readlines()]
        return db

    def loop_for_write():

        for sentences in title:
            lst = random_word_choice(sentences)
            if lst[0] != "err":
                answer = lst[1]
                text = lst[0]
                add_task(text, answer)

    for path in os.listdir(PATH_TEXTS):
        if path.endswith(".sents"):
            title = open_text()
            loop_for_write()


def main():
    read_text_dir()


if __name__ == '__main__':
    main()
