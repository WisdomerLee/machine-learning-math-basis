from transformers import MarianTokenizer, MarianMTModel, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# 1. Load tokenizer and model (e.g., Ko → En)
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 2. 예시용 데이터셋 정의 (HuggingFace Datasets API 형식)
train_data = {
    "translation": [
        {"ko": "나는 밥을 먹었다.", "en": "I ate rice."},
        {"ko": "오늘 날씨가 좋다.", "en": "The weather is good today."}
    ]
}

from datasets import Dataset
dataset = Dataset.from_list(train_data)

# 3. Preprocessing 함수
def preprocess(examples):
    inputs = [ex["ko"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# 4. TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    evaluation_strategy="no",
    save_strategy="epoch",
    deepspeed="./ds_config.json",  # ✅ DeepSpeed 사용
)

# 5. Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 6. 학습 실행
trainer.train()
