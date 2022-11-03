import torch
from transformers import BertTokenizerFast, EncoderDecoderModel


def generate_summary(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "mrm8488/bert2bert_shared-german-finetuned-summarization"
    tokenizer = BertTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
    inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

s = generate_summary("Das ist das Haus vom Nikolaus. Und nebenan das Haus vom Weihnachtsmann. Es hat große Fenster und einen Balkon. Hubert wohnt auch hier am Nordpol in direkter Nachbarschaft. Hubert ist bei der Winterweihnachtswunder GmbH angestellt.")

print(s)