from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def detect_language(text):
    text = text[:514]
    tokenizer = AutoTokenizer.from_pretrained("eleldar/language-detection")
    model = AutoModelForSequenceClassification.from_pretrained(
        "eleldar/language-detection"
    )
    ld = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = ld(text)
    return results[0]["label"], results[0]["score"]

def detect_language_formforfile(text):
    t = detect_language(text)
    return [t.__repr__()]