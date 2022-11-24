import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from pathlib import Path
import json
from numpy.typing import NDArray

class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=42, weight_path:str=None, epochs=5, device="cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.seed = seed
        self.epochs = epochs
        self.model = None
        self.labels = None
        self.device=device
        if weight_path:
            self._set_model_from_weights(weight_path)

    def _get_classes(self, y: List[str]) -> Tuple[NDArray, List[str]]:
        labels = sorted(set(y))
        ids = [i for i in range(len(labels))]
        return ids, labels

    def _set_model_from_weights(self, path:str) -> AutoModelForSequenceClassification:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path).to(self.device)
        self.labels = list(self.model.config.label2id.keys())
        
    def _tokenize(self, texts:List[str]) -> torch.Tensor:
        return self.tokenizer(texts, padding=True,
            truncation=True,
            max_length=100,
            return_tensors="pt").to(self.device)


    
    def fit(self, X:List[str], y:List[str]):
        ids, labels = self._get_classes(y)
        self.labels = labels
        id2label = dict(zip(ids,labels))
        label2id = dict(zip(labels,ids))
        X = self._tokenize(X)
        dataset = [{"input_ids": text, "label": label2id[label]} for text, label in zip(X["input_ids"],y)]
        train_ds, test_ds = train_test_split(dataset, shuffle=True, random_state=self.seed)
        batch_size = 64

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(labels), id2label=id2label, label2id=label2id
        ).to(self.device)
        weights_path="weights/bert_classifier"
        training_args = TrainingArguments(
            output_dir=weights_path,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            use_mps_device=self.device=="mps"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        model.eval()
        self.model = model

        
    
    def predict_proba(self, X:List[str]) -> NDArray:
        if self.model is None:
            raise Exception("Fit the model before inference.")
        tokens = self._tokenize(X)
        with torch.no_grad():
            logits = self.model(**tokens).logits
            return F.softmax(logits, -1).cpu().numpy()
        

    def predict(self, X:List[str])-> List[str]:
        preds = self.predict_proba(X)
        return [self.labels[i] for i in preds.argmax(-1)]


