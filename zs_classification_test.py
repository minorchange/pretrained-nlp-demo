from zs_classification import zs_class
from long_text import wiki_elefant_text


def idx_of_element(series, element):
    return series[series == element].index[0]


def test_zs_class_basic():


    sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
    
    candidate_labels = ["politics", "economy", "entertainment", "environment"]
    
    df = zs_class(candidate_labels, sequence_to_classify)
    assert sorted(list(df.scores), reverse=True) == list(df.scores), "I assume that the most probable label is on top"
    assert idx_of_element(df.labels, "politics") < idx_of_element(df.labels, "economy")
    assert idx_of_element(df.labels, "economy") < idx_of_element(df.labels, "entertainment")


def test_zs_class_long_text():

    sequence_to_classify = wiki_elefant_text
    
    candidate_labels = ["elefant", "lion", "frog", "environment", "Rakete", "Transaktion"]
    
    df = zs_class(candidate_labels, sequence_to_classify)
    assert sorted(list(df.scores), reverse=True) == list(df.scores), "I assume that the most probable label is on top"
    assert idx_of_element(df.labels, "elefant") < idx_of_element(df.labels, "lion")
    assert idx_of_element(df.labels, "lion") < idx_of_element(df.labels, "frog")
    


# test_zs_class()
# test_zs_class_long_text()