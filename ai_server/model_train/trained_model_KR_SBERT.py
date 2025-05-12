from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import kss

# 학습된 KR-SBERT 모델 로드 (모델은 가수, 성악가, 보컬트레이너, 뮤지컬 배우 등으로 학습됨)
model = SentenceTransformer("kr_sbert_finetuned")

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"[^\w\s.,]", "", text)
    return text.strip()

def get_mean_embedding(text):
    text = clean_text(text)
    sentences = kss.split_sentences(text) or [text]
    embeddings = model.encode(sentences)
    return np.mean(embeddings, axis=0)

# 예시 이력서 (음악 분야와 비음악 분야를 구분)
resumes = [
    (
        "홍길동 | 27세 | 서울 거주\n"
        "서울대학교 음악대학 성악 전공 학사\n"
        "다양한 무대 경험: 오페라, 뮤지컬, 콘서트 등\n"
        "주요 역량: 훌륭한 발성, 표현력, 무대 매너\n"
        "경력: 국내외 공연 참여 및 경연 대회 수상 경력 보유\n"
    ),
    (
        "이영희 | 30세 | 부산 거주\n"
        "국제보컬 아카데미 수료, 보컬 트레이닝 전문 자격증 소지\n"
        "다양한 보컬 워크샵 및 트레이닝 경력 풍부\n"
        "주요 역량: 보컬 지도, 음성 분석 및 피드백 제공\n"
        "경력: 유명 엔터테인먼트 회사와의 협업 경험\n"
    ),
    (
        "박철수 | 32세 | 대구 거주\n"
        "컴퓨터 공학 학사\n"
        "5년 이상의 소프트웨어 개발 및 데이터 분석 경험\n"
        "주요 기술: 파이썬, 머신러닝, 데이터베이스\n"
        "경력: IT 스타트업에서 프로젝트 리딩 경험\n"
    )
]

# 예시 채용공고 (음악 관련과 IT 관련 공고를 포함)
job_listings = [
    "서울 음악원 - 성악 강사 모집\n필수 조건: 성악 전공 및 공연 경력 필수, 학생 대상 발성 지도 경험 우대",
    "K-POP 엔터테인먼트 - 보컬 트레이너 채용\n전제 조건: 보컬 트레이닝 및 음성 분석 경험, 전문 자격증 소지자 우대",
    "뮤지컬 제작사 - 뮤지컬 배우 모집\n요구 조건: 무대 경험과 연기력, 뮤지컬 및 콘서트 공연 경력 우대",
    "ABC 테크 - 소프트웨어 엔지니어 채용\n필수 조건: 5년 이상의 백엔드 개발 경험, 파이썬 및 클라우드 환경 경험",
    "XYZ 데이터 - 데이터 분석가 모집\n우대 조건: 머신러닝 및 통계 분석 경험, 석사 학위 소지자"
]

# 텍스트 정제: 각 이력서와 채용공고에 대해 불필요한 문자를 제거합니다.
resumes_cleaned = [clean_text(resume) for resume in resumes]
job_listings_cleaned = [clean_text(job) for job in job_listings]

# 임베딩 계산: 전처리된 텍스트로부터 평균 임베딩 벡터를 구합니다.
resume_embeddings = np.array([get_mean_embedding(resume) for resume in resumes_cleaned])
job_embeddings = np.array([get_mean_embedding(job) for job in job_listings_cleaned])

# 각 이력서와 모든 채용공고 간의 유사도를 계산 및 출력
for i, resume_embedding in enumerate(resume_embeddings):
    similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]
    sorted_indices = similarity_scores.argsort()[::-1]  # 유사도 내림차순 정렬

    print(f"\n이력서 {i + 1}와 가장 유사한 채용 공고:")
    # 각 이력서와 채용공고의 유사도 점수와 내용을 출력
    for rank, idx in enumerate(sorted_indices, start=1):
        print(f"{rank}. [Score: {similarity_scores[idx]:.4f}] {job_listings[idx]}")
