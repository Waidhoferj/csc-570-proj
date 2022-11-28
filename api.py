from flask import Flask, jsonify, request
from sklearn.pipeline import Pipeline
from classifiers.mlp import MajorMlpClassifier
from classifiers.bert import BertClassifier
from embeddings.bert import BertSentenceEmbedder
from helper import get_recommendations
import os
from typing import List
from flask_cors import CORS

pipeline:BertClassifier
labels:List[str]
app= Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

def load_mlp_pipeline(device="cpu") -> Pipeline:
    embedder = BertSentenceEmbedder(device, padding_length=1000)
    classifier = MajorMlpClassifier(device)
    classifier.load_weights("weights/major_classifier")
    return Pipeline(steps=[
        ("Phrase Embedder", embedder),
        ("Embedding Classifier", classifier)
    ])


    





@app.route("/recommend", methods=["POST"])
def get_major_recommendations():
    if request.method=='POST':
        posted_data = request.get_json()
        description = posted_data['query']
        probs = pipeline.predict_proba(description)
        recommendations = get_recommendations(probs, labels, n=3)[0]
        return jsonify(list(recommendations))

    
#  main thread of execution to start the server
if __name__=='__main__':
    pipeline = BertClassifier(device="mps")
    weights_path = os.path.join("weights", "bert_classifier_deployment_weights")
    pipeline.load_weights(weights_path)
    labels = pipeline.labels
    app.run(debug=True)