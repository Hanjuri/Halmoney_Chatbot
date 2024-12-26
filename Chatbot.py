from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

API_KEY = "sk-your-openai-api-key"

# FastAPI 앱 생성
app = FastAPI()

class UserMessage(BaseModel):
    message: str
    history: list  # 이전 대화 기록

class CustomOpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat(self, model: str, messages: list):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.to_dict()

client = CustomOpenAIClient(api_key=API_KEY)

@app.post("/chat")
async def chat_endpoint(user_message: UserMessage):
    try:
        # OpenAI API 호출
        response = client.chat(
            model="gpt-4",
            messages=user_message.history + [{"role": "user", "content": user_message.message}]
        )
        assistant_message = response["choices"][0]["message"]["content"]
        return {"assistant_message": assistant_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")