from transformers import MarianTokenizer, MarianMTModel


def translate(src_lang, tgt_lang, text_to_translate):
    
    text_to_translate_cropped = text_to_translate[:512]
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    batch = tokenizer([text_to_translate_cropped], return_tensors="pt")
    generated_ids = model.generate(**batch)
    translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if len(text_to_translate) > 512:
        translated_text += " ..."
    return translated_text
