from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd


def named_entity_recognition_old(text):
    # model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
    model_name = "Davlan/xlm-roberta-large-ner-hrl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
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


def named_entity_recognition(text):
    
    # Make it cope with arbitrarily long input by processing sentence by sentence. 
    # NER does not profit a lot from sentence spanning information. 

    textlist = text.split(".")
    textlist = [t.strip(" ") for t in textlist]

    model_name = "Davlan/xlm-roberta-large-ner-hrl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    ner_results = nlp(textlist)

    flat_list = [item for sublist in ner_results for item in sublist]

    locs, pers, orgs = [], [], []

    for e in flat_list:
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