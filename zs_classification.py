# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2Tokenizer
from transformers import pipeline
import pandas as pd


def zs_class_minimal(candidate_labels, sequence_to_classify):

    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    df = pd.DataFrame({k:output[k] for k in ('labels', 'scores') if k in output})
    return(df)


# def zs_class_trunc(candidate_labels, sequence):

#     model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # tokenizer = DebertaV2Tokenizer

#     sequence = "Bananen sind sehr lecker. Aber sind sie auch schmackhafter als Pizza?"
#     # hypothesis = "Emmanuel Macron is the President of France"

#     # input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
#     input = tokenizer(sequence, truncation=True, return_tensors="pt")  # truncate to 512, return pytorch tensors
#     device = "cpu"  # device = "cuda:0" or "cpu"
#     output = model(input["input_ids"].to(device)) 
#     prediction = torch.softmax(output["logits"][0], -1).tolist()

#     # candidate_labels = ["Frucht", "Banana", "Banane", "Auto", "Essen"]
#     prediction = {
#         name: round(float(pred) * 100, 1) for pred, name in zip(prediction, candidate_labels)
#     }

#     print(prediction)


zs_class = zs_class_minimal