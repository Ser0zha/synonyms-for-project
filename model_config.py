NAME = "ai-forever/rut5-base"
INPUT_SIZE = 128
OUTPUT_SIZE = 4


def prompt(sentence: str, word: str) -> str:
    return f'Does the word "{word}" fit in the gap: "{sentence}"?'

