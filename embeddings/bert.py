import torch
from transformers import BertModel, BertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BertSentenceEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, device="cpu",padding_length=50):
        """
        Args:
            `device`: pytorch device for inference. Either 'cpu' or a specific type of GPU.
            `padding_length`: The max sentence token length. Shorter sentences are padded to this length.
        """
        self._device = device
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self._model = model.to(device)
        self._model.eval()
        self._padding_length = padding_length

    def transform(self, X:list) -> np.ndarray:
        """
        Transforms sentences into embeddings

        Args:
            `X`: a dataset of sentences of shape (n_sentences,)
        Returns:
            Embeddings of the provided sentences of shape (n_sentences, embedding_dims)
        """
        tokens = self._tokenizer(
            X, 
            return_token_type_ids=False, 
            return_attention_mask=False,
            padding="max_length",
            truncation=True,
            max_length=self._padding_length,
            return_tensors="pt"
            )
        tokens = tokens["input_ids"].to(self._device)
        with torch.no_grad():
           hidden_states = self._model(
                            input_ids=tokens, 
                            output_hidden_states=True
                            )["hidden_states"]
        embeddings = torch.cat(hidden_states[-4:], dim=-1)
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings.cpu().numpy()



if __name__ == "__main__":
    X = [
        "Hello, my name is John.",
        "What's going on my dudes?"
    ]

    embedder = BertSentenceEmbedder("mps")
    out = embedder.transform(X)
    print(out.shape)