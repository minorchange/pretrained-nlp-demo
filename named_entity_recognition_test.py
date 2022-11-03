from named_entity_recognition import named_entity_recognition


def test_named_entity_recognition():

    text = "Das ist das Haus vom Nikolaus. Es steht am Nordpol. Die Winterwunderweihnachts GmbH dementiert das. Paul arbeitet bei der SAP."
    locs, pers, orgs = named_entity_recognition(text)

    assert "Haus vom Nikolaus" in list(locs["Ort"])
    assert "SAP" in list(orgs["Organisation"])
    assert "Paul" in list(pers["Person"])
