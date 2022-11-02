from translation import translate


def test_translation_de_en():
    text_de = "Das ist das Haus vom Nikolaus"
    text_en = translate("de", "en", text_de)
    assert "house"  in text_en

def test_translation_de_fr():
    text_de = "Das ist das Haus vom Nikolaus"
    text_fr = translate("de", "fr", text_de)
    assert "maison"  in text_fr