from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import re
import kss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI()

# OpenAI API 키 설정
client = OpenAI(api_key="YOUR_API_KEY")

# —────────────────────────────────────────────────────────────────────────——
# 1) KR-SBERT 모델 및 직무 데이터 로드 (앱 시작 시 한 번만)
kr_sbert_model = SentenceTransformer(
    "model_train\kr_sbert_finetuned"
)
# 직무 정보 CSV
job_csv_path = "career_jobs_with_detail.csv"
df_jobs = pd.read_csv(job_csv_path)

# 텍스트 전처리
def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text))
    t = re.sub(r"[^\w\s.,]", "", t)
    return t.strip()

# 문장별로 나눠 평균 임베딩 계산
def get_mean_embedding(text: str) -> np.ndarray:
    text = clean_text(text)
    sentences = kss.split_sentences(text) or [text]
    embs = kr_sbert_model.encode(sentences)
    return np.mean(embs, axis=0)

# CSV 행 하나를 프롬프트용 텍스트로 변환
def make_job_text(row) -> str:
    return (
        f"직업명: {row['직업명']}\n"
        f"요약: {row['요약설명']}\n"
        f"핵심역량: {row['핵심능력']}\n"
        f"관련자격증: {row['관련자격증']}\n"
        f"관련학과: {row['관련학과']}\n"
        f"직업훈련: {row['직업훈련']}"
    )

# 유사 직무 검색 함수
def find_similar_jobs_kr_sbert(
    query_text: str, top_k: int = 3, similarity_threshold: float = 0.5
):
    query_emb = get_mean_embedding(query_text)
    job_embs = [get_mean_embedding(txt) for txt in df_jobs.apply(make_job_text, axis=1)]
    scores = cosine_similarity([query_emb], job_embs)[0]
    sorted_idxs = scores.argsort()[::-1]
    selected = [i for i in sorted_idxs if scores[i] >= similarity_threshold][:top_k]
    if selected:
        return df_jobs.iloc[selected], selected[0], scores
    else:
        return pd.DataFrame(), None, scores

# GPT 프롬프트 생성
def build_prompt(user_resume: str, job_posting: str, target_job_texts: str) -> str:
    return f"""
당신은 사용자의 이력서를 보고, 특정 직무에 더 적합하게 만들기 위한 조언을 제공하는 경력 컨설턴트입니다.

지원자 포트폴리오:
{user_resume}

희망 채용공고:
{job_posting}

희망 직무와 관련된 실제 직무 정보는 다음과 같습니다:
{target_job_texts}

이 직무에 합격하기 위해, 사용자의 이력서에서 보완해야 할 점을 상세히 알려주세요.
"""

# —────────────────────────────────────────────────────────────────────────——
# 요청/응답 모델 정의
class MatchRequest(BaseModel):
    resume: str
    job_listings: List[str]
    job_ids: List[int]

class MatchItem(BaseModel):
    job_id: int
    score: float

class MatchResponse(BaseModel):
    results: List[MatchItem]

class CareerRequest(BaseModel):
    resume: str
    job_description: str

class CareerResponse(BaseModel):
    recommendations: List[str]

# 기존 매칭 엔드포인트 (변경 없음)
@app.post("/api/match", response_model=MatchResponse)
async def match_jobs(req: MatchRequest):
    try:
        resume_vec = get_mean_embedding(req.resume)
        job_vecs = [get_mean_embedding(txt) for txt in req.job_listings]
        scores = cosine_similarity([resume_vec], job_vecs)[0]
        top_idx = scores.argsort()[::-1][:5]

        results = [
            MatchItem(job_id=req.job_ids[i], score=round(float(scores[i] * 100), 2))
            for i in top_idx
        ]
        return MatchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 수정된 경력 추천 경로 엔드포인트
@app.post("/api/career-path", response_model=CareerResponse)
async def get_career_path(req: CareerRequest):
    """
    이력서와 채용공고(text)를 GPT로 분석하여
    경력 개발을 위한 추천 경로를 단계별로 반환합니다.
    """
    try:
        # ────────────────────────────────
        # 1) 유사 직무 검색 로직 통합
        similar_df, best_idx, sim_scores = find_similar_jobs_kr_sbert(req.job_description)
        if best_idx is not None:
            # (선택적) 로깅
            print(f"가장 유사한 직무 인덱스: {best_idx}, 명칭: {df_jobs.iloc[best_idx]['직업명']}")
        # 2) 프롬프트에 들어갈 직무 정보 텍스트 준비
        if similar_df.empty:
            target_text = (
                "해당하는 직무에 대한 정보가 데이터에 존재하지 않습니다. "
                "추가적인 직무 관련 정보를 확인해 주세요."
            )
        else:
            target_text = "\n\n".join(make_job_text(row) for _, row in similar_df.iterrows())

        # ────────────────────────────────
        # 3) GPT 프롬프트 생성
        prompt = build_prompt(req.resume, req.job_description, target_text)

        # 4) GPT API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a career consultant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()

        # 5) 줄 단위로 분리하여 리스트로 반환
        lines = [line.strip("- ") for line in answer.splitlines() if line.strip()]
        return CareerResponse(recommendations=lines)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
