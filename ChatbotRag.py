import openai
import pinecone

# OpenAI 임베딩 생성 초기화
openai.api_key = "YOUR_OPENAI_API_KEY"

# Pinecone 초기화
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")
index_name = "interview-knowledge"
pinecone.create_index(index_name, dimension=1536)  # 임베딩 차원은 모델에 따라 다름
index = pinecone.Index(index_name)

# 데이터 예시
documents = [
    {"text": "어려운 문제를 해결한 경험에 대해 설명하세요.", "metadata": {"type": "behavioral"}},
    {"text": "팀 리더십을 발휘했던 프로젝트에 대해 이야기해 보세요.", "metadata": {"type": "leadership"}},
]

# 데이터 임베딩 생성 및 저장
for doc in documents:
    embedding = openai.Embedding.create(
        input=doc["text"],
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    index.upsert([(doc["text"], embedding, doc["metadata"])])