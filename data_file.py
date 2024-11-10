import json

PATH_JSON_CORPUS = 'datasets/data_ru_lang1.json'


def open_json():
    with open(PATH_JSON_CORPUS, "r", encoding="UTF-8") as read_file:
        data_rus = json.load(read_file)
    return data_rus


# def file_splitting(fil, mode="text"):
#     lst = []
#     out = fil["zadanie1"]
#     for i in out:
#         lst.append(i[mode])
#     return lst


data_out = open_json()

# list_of_sentence = file_splitting(data_out)
#
# list_of_answer = file_splitting(data_out, mode="answer")
