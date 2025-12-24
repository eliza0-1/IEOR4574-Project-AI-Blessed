# 🌙 NutriMate — AI Recipe & Meal Planning Assistant

**NutriMate** is an AI-powered recipe generator and weekly meal-planning assistant.
It combines **RAG (Retrieval-Augmented Generation)**, **LLM-based reasoning**, and a sleek **Streamlit UI** to help users get personalized cooking inspiration — complete with calories, ingredients, steps, AI recommendations, and downloadable weekly meal plan PDFs.

## Group Member: 
| Name | UNI |
|---------|----------|
| Jing (Cassie) Du  | jd3990  |
| Xinyi (Eliza) Zhang  | xz3264  |
| Xiya Liu  | xl3592  |
| Xinting (Ginni) Wang  | xw3088  |

---

## 📦 Data & Model Artifacts (HF Hub)

* **RAG artifacts:** [https://huggingface.co/Cassiedu66/ai_blessed_recipe_generator](https://huggingface.co/Cassiedu66/ai_blessed_recipe_generator)
* **Raw recipe dataset:** [https://huggingface.co/datasets/Cassiedu66/ai-blessed_raw_recipes](https://huggingface.co/datasets/Cassiedu66/ai-blessed_raw_recipes)
    - Origin recipe data credit to: [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

---

## 📸 UI Preview

### 1. Recipe Generator
<p align="center">
  <img src="https://github.com/user-attachments/assets/069c9d91-b26f-4a1f-9ce7-42ca682d72de" width="90%" />
</p>

### 2. Weekly Meal Planner
<p align="center">
  <img src="https://github.com/user-attachments/assets/b470c8f0-5eb1-434d-945e-66456a2ee973" width="90%" />
</p>

### 3. Saved Recipes
<p align="center">
  <img src="https://github.com/user-attachments/assets/053e87fe-f5c3-42ae-b8fe-bae529a2732d" width="90%" />
</p>


---

## 🛠 Installation

### 1. Clone the repo

```bash
git clone https://github.com/Cassiedu66/IEOR4574-Project-AI-Blessed.git
cd IEOR4574-Project-AI-Blessed
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit app

```bash
streamlit run rag_streamlit.py
```

---

## 🤝 Acknowledgements

* **HuggingFace** for hosting model artifacts
* **SentenceTransformers** (MiniLM)
* **Meta Llama 3-8B Instruct** for LLM generation
* **Streamlit** for interactive UI
* Original recipe dataset contributors
