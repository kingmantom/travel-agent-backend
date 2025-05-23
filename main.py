from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field


load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = instructor.from_openai(OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
))
MODEL = "gpt-4o-mini"


df = pd.read_excel("data.xlsx")


class TripDetails(BaseModel):
    related: bool = Field(description="האם השאלה רלוונטית לטיולים?")
    region: str = Field(description="האזור בארץ. אחד מ: צפון,דרום,מרכז")
    difficulty: str = Field(description="קושי. אחד מ: קל,בינוני,קשה")
    has_water: bool


@app.post("/ask")
async def ask_route(request: Request):
    data = await request.json()
    user_question = data.get("message", "")


    trip_details = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "אתה מדריך טיולים מומחה בישראל"},
            {"role": "user", "content": user_question}
        ],
        response_model=TripDetails
    )

    print(trip_details.model_dump_json())
    if not trip_details.related:
        return {"response": "אני עוזר רק בטיולים"}
    
    filtered = df.copy()
    filtered = filtered[filtered["region"] == trip_details.region]
    filtered = filtered[filtered["has_water"] == trip_details.has_water]
    filtered = filtered[filtered["difficulty"] == trip_details.difficulty]

    if filtered.empty:
        return {"response": "לא מצאתי מסלול מתאים לפי הבקשה. אולי תנסה לשנות משהו?"}

    suggestions = "\n".join([
        f"{row['name']} – {row['תיאור קצר']} ({row['region']}, {row['difficulty']})"
        for _, row in filtered.head(3).iterrows()
    ])

    final_prompt = (
        f"המשתמש מחפש מסלול. הנה כמה הצעות:\n{suggestions}\n"
        "בחר את המומלץ ביותר והסבר למה."
    )

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "אתה מדריך טיולים מומחה בישראל."},
            {"role": "user", "content": final_prompt}
        ],
        response_model=str
    )

    return {"response": final_response}
