from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import logging
from datetime import datetime
from decouple import config

# OpenAI API 키 설정
API_KEY = config("API_KEY")
openai.api_key = API_KEY

# FastAPI 앱 생성
app = FastAPI()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 URL 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware 추가
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 요청 정보 기록
    request_body = await request.body()
    logger.info(f"Incoming Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Body: {request_body.decode('utf-8') if request_body else 'No Body'}")

    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds()

    # 응답 정보 기록
    logger.info(f"Outgoing Response: Status Code: {response.status_code}, Time: {process_time:.2f}s")
    return response


# 초기 설정 데이터 스키마
class SetupRequest(BaseModel):
    job: str
    company: str

# 요청 데이터 스키마
class ChatRequest(BaseModel):
    message: str
    history: list  # [{"role": "user", "content": "message"}]

# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    assistant_message: str
    updated_history: list

class EvaluationRequest(BaseModel):
    history: list  # [{"role": "user", "content": "message"}, ...]

class EvaluationResponse(BaseModel):
    evaluation: str

# 전역 변수로 초기 설정 정보 저장
initial_setup = {}

@app.post("/setup")
async def setup_endpoint(request: SetupRequest):
    """
    직무와 회사 정보를 설정하는 엔드포인트
    """
    try:
        global initial_setup
        initial_setup = {
            "job": request.job,
            "company": request.company
        }
        return {"message": "Initial setup completed.", "setup": initial_setup}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Chat API 엔드포인트

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print("Received /chat request:")
        print("Message:", request.message)
        print("History:", request.history)

        if not initial_setup:
            raise HTTPException(status_code=400, detail="Initial setup is required. Please call /setup first.")

        # 기존 코드
        system_message = {
            "role": "system",
            "content": (
                f"You are a helpful assistant preparing the user for a job interview. "
                f"The position is '{initial_setup['job']}' at '{initial_setup['company']}'. "
                "Ask one interview question at a time, wait for the user's answer, and then provide the next question. "
                "Please do everything in Korean."
            )
        }

        messages = [system_message] + request.history + [{"role": "user", "content": request.message}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        assistant_message = response["choices"][0]["message"]["content"]
        return ChatResponse(
            assistant_message=assistant_message,
            updated_history=messages + [{"role": "assistant", "content": assistant_message}]
        )
    except HTTPException as e:
        print("HTTP Exception:", e)
        raise e
    except Exception as e:
        print("Unexpected Error:", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(request: EvaluationRequest):
    """
    대화 내용을 기반으로 평가를 제공하는 엔드포인트
    """
    try:
        evaluation_prompt = (
            "다음은 사용자와 AI의 면접 대화 기록입니다. "
            "이 대화 내용을 분석하여 사용자의 면접 답변에 대한 전반적인 평가를 작성하고, "
            "강점과 개선해야 할 점을 포함해주세요. "
            "평가를 한국어로 작성해주세요.\n\n"
            "대화 기록:\n"
        )

        # 대화 기록을 평가 프롬프트에 추가
        for message in request.history:
            role = "사용자" if message["role"] == "user" else "AI"
            evaluation_prompt += f"{role}: {message['content']}\n"

        evaluation_prompt += "\n평가:"

        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=evaluation_prompt,
            max_tokens=500,
            temperature=0.7,
        )

        evaluation = response["choices"][0]["text"].strip()
        return EvaluationResponse(evaluation=evaluation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during evaluation: {e}")