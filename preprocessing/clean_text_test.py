from preprocessing.clean_text import clean_text


def test_clean_text():
    s = "%   Blubb $ \n Pizza."
    c = clean_text(s)
    assert c == "Blubb Pizza"
