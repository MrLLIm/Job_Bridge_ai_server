import re
import numpy as np
import kss
from sentence_transformers import SentenceTransformer

MODEL_PATH = r"C:\Users\LimWonHo\PycharmProjects\PythonProject1\model_train\kr_sbert_finetuned"
model = SentenceTransformer(MODEL_PATH)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s.,]", "", text).strip()

def get_mean_embedding(text: str) -> np.ndarray:
    txt = clean_text(text)
    sents = kss.split_sentences(txt) or [txt]
    embs = model.encode(sents)
    return np.mean(embs, axis=0)
