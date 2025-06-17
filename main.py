from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from difflib import get_close_matches
import instructor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from urllib.parse import quote

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
MODEL = "gpt-4o-mini"

try:
    df = pd.read_excel("data.xlsx")
except Exception:
    df = pd.DataFrame()

try:
    df_clustering = df.copy()
    label_cols = ["region", "difficulty", "נגישות"]
    for col in label_cols:
        df_clustering[col] = LabelEncoder().fit_transform(df_clustering[col].astype(str))

    df_clustering["has_water"] = df_clustering["has_water"].astype(int)
    df_clustering["אורך מסלול (ק\"מ)"] = pd.to_numeric(df_clustering["אורך מסלול (ק\"מ)"], errors="coerce").fillna(0)

    features = ["region", "difficulty", "נגישות", "has_water", "אורך מסלול (ק\"מ)"]
    X = df_clustering[features]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_clustering["cluster"] = kmeans.fit_predict(X)
except Exception:
    pass

def create_gmaps_link(name):
    return f"https://www.google.com/maps/search/{quote(name)}"

class TripDetails(BaseModel):
    related: bool
    region: str = ""
    difficulty: str = ""
    has_water: bool | None = None

class AccessibilityOnly(BaseModel):
    wants_accessibility: bool

def is_similar_to_greeting(text):
    greetings = ["שלום", "היי", "הי", "אהלן", "מה נשמע", "מה שלומך", "מה קורה", "בוקר טוב", "ערב טוב"]
    text = text.replace("?", "").replace(",", "").replace("!", "").strip().lower()
    return any(text.startswith(greet) for greet in greetings)

@app.post("/ask")
async def ask_route(request: Request):
    data = await request.json()
    user_question = data.get("message", "").strip()
    context = data.get("context", {})

    if is_similar_to_greeting(user_question):
        return {"response": "שלום! אני כאן כדי לעזור לך למצוא מסלולי טיול בישראל! שאל אותי על אזור, קושי, מים או כל דבר שקשור לטיולים."}

    region = context.get("region")
    difficulty = context.get("difficulty")
    has_water = context.get("has_water")
    step = context.get("step")

    # שלב הניתוח לחילוץ פרמטרים
    trip_details = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "אתה מדריך טיולים בישראל. נתח את ההודעה וחלץ אם מדובר באזור (region), קושי (difficulty) או מים (has_water)."
                " החזר רק את מה שמופיע. אל תנחש את החסר. אל תשלים מידע לבד."
            )},
            {"role": "user", "content": user_question}
        ],
        response_model=TripDetails
    )

    region = region or trip_details.region
    difficulty = difficulty or trip_details.difficulty
    has_water = has_water if has_water in [True, False] else trip_details.has_water

    if not region:
        return {"response": "באיזה אזור בארץ תרצה לטייל?", "context": {"region": "", "difficulty": difficulty, "has_water": has_water, "step": "awaiting_region"}}
    if has_water is None:
        return {"response": "האם חשוב לך שיהיו מים במסלול?", "context": {"region": region, "difficulty": difficulty, "has_water": None, "step": "awaiting_water"}}
    if not difficulty:
        return {"response": "איזה דרגת קושי תעדיף? קל, בינוני או קשה?", "context": {"region": region, "difficulty": "", "has_water": has_water, "step": "awaiting_difficulty"}}

    # אם יש followup נמשיך לשלב הנגישות
    if context.get("followup_required") or context.get("step") == "awaiting_accessibility":
        accessibility_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "משתמש נשאל האם חשובה לו נגישות לנכים. החזר JSON עם שדה wants_accessibility בלבד."},
                {"role": "user", "content": user_question},
            ],
            response_model=AccessibilityOnly
        )

        wants_accessibility = accessibility_response.wants_accessibility

        # סינון המסלולים
        filtered = df.copy()
        if region:
            filtered = filtered[filtered["region"] == region]
        if isinstance(has_water, bool):
            filtered = filtered[filtered["has_water"] == has_water]
        if difficulty:
            filtered = filtered[filtered["difficulty"] == difficulty]
        if wants_accessibility:
            filtered = filtered[filtered["נגישות"].astype(str).str.contains("נגיש", na=False)]

        if filtered.empty:
            return {"response": "לא מצאתי מסלול מתאים לפי הבקשה. אולי תנסה לשנות משהו?"}

        suggestions = "\n".join([
            f'{row["name"]} – {row["תיאור קצר"]} ({row["region"]}, {row["difficulty"]})'
            for _, row in filtered.head(3).iterrows()
        ])

        final_prompt = (
            f"המשתמש מחפש מסלול. הנה כמה הצעות:\n{suggestions}\n"
            "בחר את המסלול המומלץ ביותר.\n"
            "שלב 1: כתוב את שם המסלול במרכאות כפולות \"\" בלבד, בשורה הראשונה, בלי הסבר נוסף.\n"
            "שלב 2: לאחר מכן, הסבר מדוע בחרת בו."
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

    # אם עדיין לא שאלה על נגישות – נבקש אותה
    return {
        "response": "רק שאלה אחרונה 🙂 האם חשוב לך שהמסלול יהיה נגיש לנכים?",
        "context": {
            "followup_required": True,
            "region": region,
            "difficulty": difficulty,
            "has_water": has_water,
            "step": "awaiting_accessibility"
        }
    }

@app.post("/similar")
async def get_similar_routes(request: Request):
    data = await request.json()
    route_name = data.get("route_name")

    all_names = df_clustering["name"].astype(str).tolist()
    match = get_close_matches(route_name, all_names, n=1, cutoff=0.6)

    if not match:
        return {"response": "לא מצאתי את המסלול הזה 😕"}

    closest = match[0]
    cluster_id = df_clustering[df_clustering["name"] == closest]["cluster"].values[0]
    similar_routes = df_clustering[df_clustering["cluster"] == cluster_id]
    similar_routes = similar_routes[similar_routes["name"] != closest].head(5)

    if similar_routes.empty:
        return {"response": f"לא מצאתי מסלולים דומים ל‎{closest} 😕"}

    suggestions = "\n".join([
        f'{row["name"]} – {row["תיאור קצר"]} ({row["region"]}, {row["difficulty"]})'
        for _, row in similar_routes.iterrows()
    ])

    return {"response": f"הנה מסלולים דומים ל‎{closest}:\n{suggestions}"}
