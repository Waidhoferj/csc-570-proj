import re
import pandas as pd
import numpy as np
from typing import Tuple, List

PROGRAM = "Program"


def clean_text(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '', text_input)
    return output.lower().strip()


def get_num_courses_per_program():
    df = pd.read_csv('program_courses.csv')
    return df.groupby([PROGRAM])[PROGRAM].count()


def load_data(num_majors=20) -> Tuple[List[str], np.ndarray]:
    """
    Loads and preprocesses `course_sentences` data.
    """
    df = pd.read_csv("course_sentences.csv")
    top_majors = df.groupby("program").count().sort_values(by=["sentence"], ascending=False).head(num_majors).index
    df = df[df["program"].isin(top_majors)]
    sentences = list(df["sentence"])
    labels = np.array(df["program"])

    return sentences, labels

def get_recommendations(probs:np.ndarray, labels:List[str], n=5) -> List[List[str]]:
    """
    Args:
        `probs`: predictions array of shape (n_inputs,n_classes)
        `labels`: class labels of shape (n_classes,)
        `n`: number of recommendations
    Returns:
        Top labels based on a probability distribution
    """
    np_labels = np.array(labels)
    return np_labels[probs.argsort(-1)[:,:n]]

