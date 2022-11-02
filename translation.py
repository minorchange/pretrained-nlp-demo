# from transformers import MarianTokenizer, MarianMTModel

# src = "fr"  # source language
# trg = "en"  # target language

# model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
# model = MarianMTModel.from_pretrained(model_name)
# tokenizer = MarianTokenizer.from_pretrained(model_name)

# sample_text = "où est l'arrêt de bus ?"
# batch = tokenizer([sample_text], return_tensors="pt")

# generated_ids = model.generate(**batch)
# out =tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(out)

from transformers import MarianTokenizer, MarianMTModel


def translate(src_lang, tgt_lang, text_to_translate):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    batch = tokenizer([text_to_translate], return_tensors="pt")
    generated_ids = model.generate(**batch)
    translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translated_text