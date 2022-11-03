from nlp_tasks.language_detection.ld import detect_language


def test_detect_language():
    text_de = "Das ist das Haus vom Nikolaus."
    text_en = "And the most impotant thing: Do not smoke in the entrance hall."
    de, de_score = detect_language(text_de)
    assert de == "de"
    assert de_score > 0.95
    en, en_score = detect_language(text_en)
    assert en == "en"
    assert en_score > 0.95
