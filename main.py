from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import Trainer, TrainingArguments

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


