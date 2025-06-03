from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from difflib import SequenceMatcher, get_close_matches
import instructor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

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

# טען את הקובץ
try:
    df = pd.read_excel("data.xlsx")
except Exception as e:
    df = pd.DataFrame()
    print(f"⚠️ לא נטען קובץ הנתונים: {e}")

# Clustering על המסלולים
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
except Exception as e:
    print(f"⚠️ שגיאה ב-Clustering: {e}")

class TripDetails(BaseModel):
    related: bool = Field(description="האם השאלה רלוונטית לטיולים?")
    region: str = Field(description="האזור בארץ. אחד מ: צפון,דרום,מרכז")
    difficulty: str = Field(description="קושי. אחד מ: קל,בינוני,קשה")
    has_water: bool

class ExtraFilter(BaseModel):
    wants_accessibility: bool = Field(description="האם המשתמש רוצה שהמסלול יהיה נגיש?")
    max_length_km: float = Field(description="אורך מקסימלי של המסלול בקילומטרים", default=0.0)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def is_similar_to_greeting(text):
    greetings = ["שלום", "היי", "הי", "אהלן", "מה נשמע", "מה שלומך", "מה קורה", "בוקר טוב", "ערב טוב"]
    text = text.replace("?", "").replace(",", "").replace("!", "").strip().lower()
    return any(levenshtein_distance(text, greet) <= 1 for greet in greetings)

@app.post("/ask")
async def ask_route(request: Request):
    data = await request.json()
    user_question = data.get("message", "").strip()
    context = data.get("context", {})

    if is_similar_to_greeting(user_question):
        return {"response": "שלום! אני כאן כדי לעזור לך למצוא מסלולי טיול בישראל 🏝️ שאל אותי על אזור, קושי, מים או כל דבר שקשור לטיולים."}

    if context.get("followup_required"):
        followup_answer = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "משתמש נשאל האם חשובה לו נגישות לנכים ואורך מקסימלי של המסלול. "
                        "החזר JSON עם שני שדות: wants_accessibility (true/false), max_length_km (מספר). "
                        "אם לא חשוב לו – החזר false ו־0 בהתאמה."
                    )
                },
                {"role": "user", "content": user_question}
            ],
            response_model=ExtraFilter
        )

        trip_details = TripDetails(
            related=True,
            region=context.get("region"),
            difficulty=context.get("difficulty"),
            has_water=context.get("has_water")
        )
        extra_filters = followup_answer

        filtered = df.copy()
        if trip_details.region:
            filtered = filtered[filtered["region"] == trip_details.region]
        if isinstance(trip_details.has_water, bool):
            filtered = filtered[filtered["has_water"] == trip_details.has_water]
        if trip_details.difficulty:
            filtered = filtered[filtered["difficulty"] == trip_details.difficulty]
        if extra_filters.wants_accessibility:
            filtered = filtered[filtered["נגישות"].astype(str).str.contains("נגיש", na=False)]
        if extra_filters.max_length_km > 0:
            filtered = filtered[pd.to_numeric(filtered["אורך מסלול (ק\"מ)"], errors="coerce") <= extra_filters.max_length_km]

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

    else:
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

        if not trip_details.related:
            return {"response": "אני כאן כדי לעזור רק בטיולים 🙂 נסה לשאול על מסלול, אזור בארץ, מים או רמת קושי."}

        filled_fields = sum([
            bool(trip_details.region),
            bool(trip_details.difficulty),
            isinstance(trip_details.has_water, bool)
        ])

        if filled_fields < 2:
            return {"response": "כדי שאוכל להמליץ לך על מסלול מתאים, נסה לציין לפחות שני פרטים – אזור, רמת קושי או אם יש מים 🌊"}

        return {
            "response": "רק שאלה אחרונה 🙂 האם חשוב לך שהמסלול יהיה נגיש לנכים? ומה האורך המקסימלי שתרצה?",
            "context": {
                "followup_required": True,
                "user_question": user_question,
                "region": trip_details.region,
                "difficulty": trip_details.difficulty,
                "has_water": trip_details.has_water
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
        return {"response": f"לא מצאתי מסלולים דומים ל־{closest} 😕"}

    suggestions = "\n".join([
        f"{row['name']} – {row['תיאור קצר']} ({row['region']}, {row['difficulty']})"
        for _, row in similar_routes.iterrows()
    ])

    return {"response": f"הנה מסלולים דומים ל־{closest}:\n{suggestions}"}
