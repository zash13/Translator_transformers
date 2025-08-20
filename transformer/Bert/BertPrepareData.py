import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def bert_data(max_len, file_name="train.csv"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_BASE_PATH = os.path.join(
        BASE_DIR, "..", "datasets", "QuoraQuestionPairs", ""
    )
    TRAIN_DATASET_PATH = os.path.normpath(os.path.join(DATASET_BASE_PATH, file_name))

    df = pd.read_csv(TRAIN_DATASET_PATH)
    """
    print(df.info())
    print(df.head(2))
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 404290 entries, 0 to 404289
    Data columns (total 6 columns):
     #   Column        Non-Null Count   Dtype 
    ---  ------        --------------   ----- 
     0   id            404290 non-null  int64 
     1   qid1          404290 non-null  int64 
     2   qid2          404290 non-null  int64 
     3   question1     404289 non-null  object
     4   question2     404288 non-null  object
     5   is_duplicate  404290 non-null  int64 
    dtypes: int64(4), object(2)
    memory usage: 18.5+ MB
    None
       id  qid1  qid2                                          question1                                          question2  is_duplicate
    0   0     1     2  What is the step by step guide to invest in sh...  What is the step by step guide to invest in sh...             0
    1   1     3     4  What is the story of Kohinoor (Koh-i-Noor) Dia...  What would happen if the Indian government sto...             0
    """
    df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)
    df["question1"] = df["question1"].astype(str)
    df["question2"] = df["question2"].astype(str)
    labels = df["is_duplicate"].astype(int).values

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    encodings = tokenizer(
        list(df["question1"]),
        list(df["question2"]),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="np",  # numpy, not tf
    )

    return encodings["input_ids"], encodings["token_type_ids"], labels, vocab_size
