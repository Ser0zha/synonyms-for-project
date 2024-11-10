# from transformers import T5Tokenizer, T5ForConditionalGeneration
#
# # Инициализация модели и токенизатора
# model_name = "ai-forever/rut5-base"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
#
#
# def generate_correct_word(text):
#     # Формируем текст запроса для модели
#     input_text = text.replace("WORD", "<extra_id_0>")
#     inputs = tokenizer(input_text, return_tensors="pt")
#
#     # Генерируем ответ
#     outputs = model.generate(**inputs, max_length=10, num_beams=5, early_stopping=True)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     return generated_text
#
#
# # Пример использования функции
# student_text = "В WORD известного детского писателя входят рассказы и сказки о природе и животных."
# student_answer = "книгу"  # Ответ студента
#
# # Генерируем ответ модели
# model_answer = generate_correct_word(student_text)
#
# # Проверяем ответ студента
# if model_answer.strip() == student_answer.strip():
#     print("Ответ студента верный!")
# else:
#     print("Ответ студента неверный.")
#     print(f"Правильный ответ: {model_answer}")

import torch
print(torch.__version__)  # Проверка версии PyTorch
print(torch.cuda.is_available())  # Проверка доступности CUDA
print(torch.cuda.get_device_name(0))  # Проверка имени GPU
