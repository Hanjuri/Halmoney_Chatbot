from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from decouple import config


# OpenAI API 키 설정
API_KEY = config("API_KEY")
openai.api_key = API_KEY

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 URL 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 데이터 스키마
class UserMessage(BaseModel):
    message: str
    history: list  # [{"role": "user", "content": "message"}]

# OpenAI API 호출 함수
class ChatRequest(BaseModel):
    message: str
    history: list  # [{"role": "user", "content": "message"}, ...]

# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    assistant_message: str
    updated_history: list

# Chat API 엔드포인트
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 시스템 메시지 정의
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in job interview preparation for learning material teachers. "
                "Ask one interview question at a time, wait for the user's answer, and then provide the next question. "
                "Continue this process for a question tailored to the learning material teacher profession. "
                "Please do everything in Korean."
            )
        }

        # 메시지 구성
        messages = [system_message] + request.history + [{"role": "user", "content": request.message}]

        # OpenAI Chat API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )

        # AI의 응답 추출
        assistant_message = response["choices"][0]["message"]["content"]

        # 응답과 업데이트된 히스토리 반환
        return ChatResponse(
            assistant_message=assistant_message,
            updated_history=messages + [{"role": "assistant", "content": assistant_message}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")