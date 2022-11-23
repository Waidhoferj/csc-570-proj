from classifiers.mlp import ProgramClassifier
from embeddings.bert import BertSentenceEmbedder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

device = "mps"
def test():
    df = pd.read_csv("course_sentences.csv")
    embedder = BertSentenceEmbedder(device, padding_length=1000)
    embeddings = embedder.transform(list(df["sentence"]))
    labels = df["program"]
    seed = 2
    x_train, x_test, y_train,y_test = train_test_split(embeddings, labels, random_state=seed, shuffle=True)
    classifier = ProgramClassifier(device)
    knn = KNeighborsClassifier(n_neighbors=10)
    mlp = MLPClassifier([512,256,128])

    # print("Sklearn MLP score", np.mean(cross_val_score(mlp, x_train, y_train, cv=5)))
    # print("Classifier score", np.mean(cross_val_score(classifier, x_train, y_train, cv=5)))

    classifier.fit(x_train,y_train)
    preds = classifier.predict(x_train)
    
    prompt = "I really like artificial intelligence and data structures."
    embedded_prompt = embedder.transform([prompt])
    result = classifier.predict(embedded_prompt)
    print(prompt,result)



if __name__ == "__main__":
    test()
