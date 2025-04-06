from transformers import MarianMTModel, MarianTokenizer, pipeline

# 예: 영어 → 한국어 번역
model_name = "Helsinki-NLP/opus-mt-en-ko"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translation = pipeline("translation_en_to_ko", model=model, tokenizer=tokenizer)
result = translation("How are you today?")
print(result[0]['translation_text'])  # 출력 예: 오늘 기분이 어떠세요?
