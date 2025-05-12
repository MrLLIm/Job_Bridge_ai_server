from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# 모델 불러오기
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 데이터 로드
df = pd.read_csv(r"kr_sbert_dataset_careernet.csv.csv", encoding="utf-8-sig")

# float 타입 방지: 모두 문자열로 변환
df["sentence1"] = df["sentence1"].astype(str)
df["sentence2"] = df["sentence2"].astype(str)

# 3. Sentence-BERT 형식의 InputExample 데이터 변환
train_examples = [
    InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["label"])
    for _, row in df.iterrows()
]

# DataLoader 생성
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

# 손실 함수 설정
train_loss = losses.CosineSimilarityLoss(model)

# 모델 학습
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=100, warmup_steps=100)

# 7. 모델 저장
model.save("kr_sbert_finetuned")
print("KR-SBERT 도메인 적응 미세조정 완료!")
