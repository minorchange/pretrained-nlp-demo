from ts import similarity_matrix
import numpy as np


def test_similarity_matrix():

    texts1 = [
        "Die Katze sitzt draußen",
        "Ein Mann spielt Gitarre",
        "Der neue Film ist großartig",
    ]

    texts2 = [
        "Der Hund spielt im Garten",
        "Eine Frau sieht fern",
        "Der neue Film ist so toll",
        "Der neue Film ist so toll. Ich freue mich.",
    ]

    sm = similarity_matrix(texts1, texts2)

    assert np.max(sm) == sm[2, 2]
    assert sm[2, 3] > sm[0, 0]
    assert sm[2, 3] > sm[0, 1]
