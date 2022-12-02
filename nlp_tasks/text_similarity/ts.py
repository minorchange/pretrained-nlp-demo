from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def cosine_similarity(A, B):
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return cosine


def similarity_matrix(texts1, texts2):
    model = SentenceTransformer("Sahajtomar/German-semantic")
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    sim_mat = -1 * np.ones([len(texts1), len(texts2)])

    for i in range(len(texts1)):
        for j in range(len(texts2)):
            sim_mat[i, j] = cosine_similarity(embeddings1[i], embeddings2[j])

    return sim_mat


def visualize_similarities(similarities, lables1, labels2):

    sim = np.round(similarities, 2)
    fig, ax = plt.subplots()
    im = ax.imshow(sim, cmap="winter")

    ts1 = [s[:40] for s in lables1]
    ts2 = [s[:40] for s in labels2]

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(ts2)), labels=ts2)
    ax.set_yticks(np.arange(len(ts1)), labels=ts1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ts1)):
        for j in range(len(ts2)):
            text = ax.text(j, i, sim[i, j], ha="center", va="center", color="w")

    ax.set_title("Text Similarity")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":

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

    visualize_similarities(sm, texts1, texts2)
    print()
