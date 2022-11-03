from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd


def named_entity_recognition(text):
    tokenizer = AutoTokenizer.from_pretrained(
        "Davlan/bert-base-multilingual-cased-ner-hrl"
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "Davlan/bert-base-multilingual-cased-ner-hrl"
    )
    nlp = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    ner_results = nlp(text)

    locs, pers, orgs = [], [], []

    for e in ner_results:
        if e["entity_group"] == "ORG":
            orgs += [[e["word"], e["score"]]]
        if e["entity_group"] == "PER":
            pers += [[e["word"], e["score"]]]
        if e["entity_group"] == "LOC":
            locs += [[e["word"], e["score"]]]

    locs = pd.DataFrame(locs, columns=["Ort", "Score"])
    pers = pd.DataFrame(pers, columns=["Person", "Score"])
    orgs = pd.DataFrame(orgs, columns=["Organisation", "Score"])

    return locs, pers, orgs
