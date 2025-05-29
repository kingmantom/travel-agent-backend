from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import instructor
from pydantic import BaseModel, Field

app = FastAPI()

# ✨ מאפשר גישה מה-Frontend שב-Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = instructor.from_openai(OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
))
MODEL = "gpt-4o-mini"

# טוען את הנתונים מקובץ Excel
df = pd.read_excel("data.xlsx")

# מבנה הבקשה מהמודל
class TripDetails(BaseModel):
    related: bool = Field(description="האם השאלה רלוונטית לטיולים?")
    region: str = Field(description="האזור בארץ. אחד מ: צפון,דרום,מרכז")
    difficulty: str = Field(description="קושי. אחד מ: קל,בינוני,קשה")
    has_water: bool

@app.post("/ask")
async def ask_route(request: Request):
    data = await request.json()
    user_question = data.get("message", "").strip()

    # 🟢 תוספת – בדיקת ברכה בלבד
    greetings = ["שלום", "היי", "הי", "אהלן", "מה נשמע", "מה שלומך", "מה קורה", "בוקר טוב", "ערב טוב"]
    if user_question.lower() in [g.lower() for g in greetings]:
        return {
            "response": "שלום! אני כאן כדי לעזור לך למצוא מסלולי טיול בישראל 🏝️ שאל אותי על אזור, קושי, מים או כל דבר שקשור לטיולים."
        }

    # חילוץ פרטי הטיול מתוך השאלה
    trip_details = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "אתה מדריך טיולים מומחה בישראל. "
                    "המטרה היא לחלץ מהשאלה של המשתמש שלושה פרמטרים: "
                    "אזור בארץ (region), רמת קושי (difficulty), והאם יש מים במסלול (has_water). "
                    "אם לא נאמר מספיק מידע, אבל מדובר בתחום הטיולים – קבע related=True, אבל השאר את הערכים ריקים. "
                    "אם זו לא שאלה שקשורה בכלל לטיולים – קבע related=False בלבד."
                )
            },
            {"role": "user", "content": user_question}
        ],
        response_model=TripDetails
    )

    # שאלה לא קשורה בכלל לטיולים
    if not trip_details.related:
        return {"response": "אני כאן כדי לעזור רק בטיולים 🙂 נסה לשאול על מסלול, אזור בארץ, מים או רמת קושי."}

    # אם פחות מ-2 פרטים מולאו
    filled_fields = sum([
        bool(trip_details.region),
        bool(trip_details.difficulty),
        isinstance(trip_details.has_water, bool)
    ])

    if filled_fields < 2:
        return {"response": "כדי שאוכל להמליץ לך על מסלול מתאים, נסה לציין לפחות שני פרטים – אזור, רמת קושי או אם יש מים 🌊"}

    # סינון לפי הקריטריונים
    filtered = df.copy()
    if trip_details.region:
        filtered = filtered[filtered["region"] == trip_details.region]
    if isinstance(trip_details.has_water, bool):
        filtered = filtered[filtered["has_water"] == trip_details.has_water]
    if trip_details.difficulty:
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

    return {"response": str(final_response)}