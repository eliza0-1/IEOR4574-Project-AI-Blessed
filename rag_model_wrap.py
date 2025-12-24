# ============================
# rag_model.py (FINAL VERSION)
# ============================

import os
import pickle
import pandas as pd
import numpy as np
import re
import ast
from typing import List, Dict, Any
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI
from huggingface_hub import hf_hub_download

# -------------------------
# HuggingFace Llama client
# -------------------------
os.environ['HF_TOKEN'] = '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'

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Global placeholders — these will be populated after load_rag_state()
recipes = None
embeddings = None
embedder = None
nearest_index = None

# =====================================================================
#                  PREPROCESS UTILITIES (EXACTLY YOURS)
# =====================================================================

def parse_list_like(cell):
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        return []


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Your original preprocessing logic wrapped into a function."""

    recipes = df.copy()

    # Parse tags/ingredients/steps
    for col in ["tags", "ingredients", "steps"]:
        recipes[col] = recipes[col].apply(parse_list_like)

    recipes["tags_clean"] = recipes["tags"].apply(lambda t: [str(x).lower() for x in t])
    recipes["ingredients_clean"] = recipes["ingredients"].apply(lambda t: [str(i).lower() for i in t])
    recipes["steps_clean"] = recipes["steps"].apply(lambda t: [str(s) for s in t])

    recipes["calories"] = pd.to_numeric(recipes["calories"], errors="coerce")
    recipes["minutes"] = pd.to_numeric(recipes["minutes"], errors="coerce")
    recipes.dropna(subset=["name", "calories"], inplace=True)

    # Serving extraction
    SERVING_PATTERN = re.compile(r"(?:serves|serve|serving(?:s)?|makes)\s+(\d+)")

    def infer_servings(steps: List[str]):
        text = " ".join(steps).lower()
        match = SERVING_PATTERN.search(text)
        return float(match.group(1)) if match else np.nan

    recipes["servings_hint"] = recipes["steps_clean"].apply(infer_servings)

    # Build retrieval text
    def build_retrieval_text(row):
        tag_text = ", ".join(row["tags_clean"])
        ing_text = ", ".join(row["ingredients_clean"])
        step_text = " | ".join(row["steps_clean"])
        serving_text = f" | servings: {int(row['servings_hint'])}" if pd.notna(row["servings_hint"]) else ""
        return (
            f"name: {row['name']} | tags: {tag_text} | ingredients: {ing_text} | "
            f"steps: {step_text} | calories: {row['calories']:.1f}{serving_text}"
        )

    recipes["retrieval_text"] = recipes.apply(build_retrieval_text, axis=1)
    recipes = recipes.reset_index(drop=True)

    return recipes


# =====================================================================
#         STEP 1: BUILD + SAVE RAG MODEL (RUN ONCE)
# =====================================================================
def build_and_save_rag(csv_path="RAW_recipes.csv", artifact_dir="rag_artifacts"):
    print("📥 Loading dataset...")
    df = pd.read_csv(csv_path)

    print("🔧 Preprocessing...")
    recipes = preprocess_data(df)

    print("🧠 Loading embedder (MiniLM)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚙️ Encoding recipes (this takes time)...")
    texts = recipes["retrieval_text"].tolist()
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("🔎 Building nearest neighbors index...")
    index = NearestNeighbors(metric="cosine", n_neighbors=25)
    index.fit(embeddings)

    print("💾 Saving artifacts...")
    os.makedirs(artifact_dir, exist_ok=True)

    recipes.to_pickle(f"{artifact_dir}/recipes.pkl")
    np.save(f"{artifact_dir}/embeddings.npy", embeddings)

    with open(f"{artifact_dir}/index.pkl", "wb") as f:
        pickle.dump(index, f)

    embedder.save(f"{artifact_dir}/embedder")

    print("✅ DONE — RAG model saved to 'rag_artifacts/'")


# =====================================================================
#         STEP 2: LOAD RAG MODEL (FAST) — USED BY STREAMLIT
# =====================================================================
# def load_rag_state(artifact_dir="rag_artifacts"):
#     global recipes, embeddings, embedder, nearest_index

#     recipes = pd.read_pickle(f"{artifact_dir}/recipes.pkl")
#     embeddings = np.load(f"{artifact_dir}/embeddings.npy")

#     with open(f"{artifact_dir}/index.pkl", "rb") as f:
#         nearest_index = pickle.load(f)

#     embedder = SentenceTransformer(f"{artifact_dir}/embedder")

#     print("✅ RAG model loaded successfully")
#     return recipes, embeddings, embedder, nearest_index

def load_rag_state():
    global recipes, embeddings, embedder, nearest_index
    
    repo_id = "Cassiedu66/ai_blessed_recipe_generator"

    print("📥 Downloading RAG artifacts from HuggingFace Hub...")

    # Download index + data
    recipes_path = hf_hub_download(repo_id, "rag_artifacts/recipes.pkl")
    embeddings_path = hf_hub_download(repo_id, "rag_artifacts/embeddings.npy")
    index_path = hf_hub_download(repo_id, "rag_artifacts/index.pkl")

    # Load them
    recipes = pd.read_pickle(recipes_path)
    embeddings = np.load(embeddings_path)

    with open(index_path, "rb") as f:
        nearest_index = pickle.load(f)

    # ------------------------------------------------------------
    # Load embedder freshly from HF (not from artifacts)
    # ------------------------------------------------------------
    print("📥 Loading embedder model from HF...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("✅ RAG model fully loaded from HuggingFace Hub")
    return recipes, embeddings, embedder, nearest_index




# =====================================================================
#                     RETRIEVAL + FILTERING LOGIC
# =====================================================================

# Cuisine keywords (from your code)
TAG_FREQ = Counter()
CUISINE_KEYWORDS = {
    'mexican', 'italian', 'indian', 'chinese', 'thai', 'french', 'japanese',
    'greek', 'mediterranean', 'spanish', 'korean', 'vietnamese', 'caribbean',
    'german', 'lebanese', 'turkish', 'moroccan', 'ethiopian', 'american',
    'middle eastern', 'latin', 'southwestern', 'cajun'
}


def extract_constraints(question: str) -> Dict[str, Any]:
    q_lower = question.lower()
    constraints = {
        "tag_keywords": set(),
        "max_calories": None,
        "include_ingredients": set(),
        "exclude_ingredients": set(),
        "max_minutes": None,
    }

    cal_match = re.search(r"(?:calories?|kcals?).*?(\d+)", q_lower)
    if cal_match:
        constraints["max_calories"] = float(cal_match.group(1))

    time_match = re.search(r"(\d+)\s*(?:minutes|mins|min)", q_lower)
    if time_match:
        constraints["max_minutes"] = float(time_match.group(1))

    for keyword in CUISINE_KEYWORDS:
        if keyword in q_lower:
            constraints["tag_keywords"].add(keyword)

    include_hits = re.findall(r"with ([a-z\s]+?)(?:,| and | but |\.|$)", q_lower)
    constraints["include_ingredients"] = {hit.strip() for hit in include_hits}

    exclude_hits = re.findall(r"(?:without|no) ([a-z\s]+?)(?:,| and | but |\.|$)", q_lower)
    constraints["exclude_ingredients"] = {hit.strip() for hit in exclude_hits}

    return constraints


def apply_structured_filters(df: pd.DataFrame, constraints: Dict[str, Any]):
    filtered = df

    if constraints["max_calories"]:
        filtered = filtered[filtered["calories"] <= constraints["max_calories"]]

    if constraints["max_minutes"]:
        filtered = filtered[filtered["minutes"] <= constraints["max_minutes"]]

    if constraints["tag_keywords"]:
        filtered = filtered[
            filtered["tags_clean"].apply(
                lambda tags: any(kw in tags for kw in constraints["tag_keywords"])
            )
        ]

    if constraints["include_ingredients"]:
        filtered = filtered[
            filtered["ingredients_clean"].apply(
                lambda ings: all(
                    any(need in ing for ing in ings)
                    for need in constraints["include_ingredients"]
                )
            )
        ]

    if constraints["exclude_ingredients"]:
        filtered = filtered[
            filtered["ingredients_clean"].apply(
                lambda ings: not any(
                    any(ban in ing for ing in ings)
                    for ban in constraints["exclude_ingredients"]
                )
            )
        ]

    return filtered


def retrieve_recipes(question: str, top_k=5, fetch_k=40):
    """Retrieval using LOADED global variables."""
    global recipes, embedder, nearest_index

    constraints = extract_constraints(question)

    query_vec = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = nearest_index.kneighbors(query_vec, n_neighbors=fetch_k)

    candidate_df = recipes.iloc[indices[0]].copy()
    candidate_df["similarity"] = 1 - distances[0]

    filtered = apply_structured_filters(candidate_df, constraints)
    return filtered.head(top_k) if not filtered.empty else candidate_df.head(top_k)


# =====================================================================
#                      PROMPT + GENERATION LOGIC
# =====================================================================

SYSTEM_PROMPT = (
    "You are a culinary assistant that recommends recipes based on structured "
    "RAW_recipes data. Always respect hard filters (calories, ingredients, cuisines) "
    "before answering. When context does not satisfy constraints, explain the limitation."
)


def format_context(recs: List[Dict[str, Any]]):
    blocks = []
    for idx, rec in enumerate(recs, start=1):
        block = [
            f"Recipe {idx}: {rec['name']}",
            f"Tags: {', '.join(rec['tags'][:10])}",
            f"Calories: {rec['calories']} (minutes: {rec['minutes']})",
            f"Ingredients: {', '.join(rec['ingredients'][:12])}",
            f"Steps: {' | '.join(rec['steps'])}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_prompt(question: str, retrieved: List[Dict[str, Any]]):
    context_block = format_context(retrieved) if retrieved else "(no recipes matched)"
    user_message = (
        "You are given structured recipe candidates. "
        "Use them to answer the user's question with concrete names, ingredients, steps, "
        "and calories. If calories or other constraints are provided, explicitly confirm "
        "they are satisfied.\n\n"
        f"User question: {question}\n"
        f"Recipe context:\n{context_block}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def rag_answer(question: str, top_k=5):
    retrieved_df = retrieve_recipes(question, top_k=top_k)
    retrieved = retrieved_df.to_dict("records")

    messages = build_prompt(question, retrieved)

    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )

    answer = resp.choices[0].message.content
    return {"question": question, "retrieved": retrieved, "answer": answer}


def preview_retrieved(recipes_for_prompt):
    rows = []
    for rec in recipes_for_prompt:
        rows.append({
            "name": rec["name"],
            "calories": rec["calories"],
            "tags": ", ".join(rec["tags"][:8]),
            "ingredients": ", ".join(rec["ingredients"][:8]),
        })
    return pd.DataFrame(rows)
