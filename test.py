from classifiers.mlp import MajorMlpClassifier
from embeddings.bert import BertSentenceEmbedder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from classifiers.bert import BertClassifier
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from helper import load_data, get_recommendations, plot_confusion_matrix
import matplotlib.pyplot as plt
import os

device = "mps"

def evaluate(load_weights=True):
    """
    Performs basic train/test split evaluation. 
    """
    os.makedirs("figures", exist_ok=True)
    sentences, labels = load_data()
    embedder = BertSentenceEmbedder(device, padding_length=1000)
    
    seed = 2
    x_train, x_test, y_train,y_test = train_test_split(sentences, labels, random_state=seed, shuffle=True)
    train_embeddings = embedder.transform(x_train)
    test_embeddings = embedder.transform(x_test)
    sklearn_mlp = MLPClassifier([512,256,128])
    mlp = MajorMlpClassifier(device)
    bert_classifier = BertClassifier(device=device, weight_path="./weights/program_classifier/checkpoint-3400"if load_weights else None)

    if not load_weights:
        bert_classifier.fit(x_train,y_train)
    mlp.fit(train_embeddings,y_train)
    sklearn_mlp.fit(train_embeddings, y_train)
    classes = bert_classifier.labels

    def report(name, classifier, x,y):
        preds = classifier.predict(x)
        print(name)
        print(classification_report(y, preds))
        plot_confusion_matrix(y,preds, classes)
        plt.savefig(f"figures/{name}_cm.png")
        plt.clf()

    report("bert_classifier",bert_classifier, x_test, y_test)
    report("sklearn_mlp",sklearn_mlp, test_embeddings, y_test)
    report("major_mlp",mlp, test_embeddings, y_test)

    
    


def demo():
    """
    Interact with a model on the command line.
    """
    bert_classifier = BertClassifier(weight_path="./weights/program_classifier/checkpoint-3400")

    while True:
        command = input("Describe your ideal major: ")
        if command.lower() == "q" or command.lower() == "quit":
            break
        probs = bert_classifier.predict_proba(command)
        labels = bert_classifier.labels
        print(get_recommendations(probs, labels, n=3)[0])




if __name__ == "__main__":
    demo()
