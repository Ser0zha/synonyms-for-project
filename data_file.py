import json

address1 = "data_ru_lang.json"


def open_json(address: str):
    with open(address, "r", encoding="UTF-8") as read_file:
        data_rus = json.load(read_file)
    return data_rus


def file_splitting(fil, mode="text"):
    lst = []
    out = fil["zadanie1"]
    for i in out:
        lst.append(i[mode])
    return lst


data_out = open_json(address1)

list_of_sentence = file_splitting(data_out)

list_of_answer = file_splitting(data_out, mode="answer")
