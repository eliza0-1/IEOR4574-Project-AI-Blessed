import streamlit as st
from datetime import datetime, timedelta
import json
from fpdf import FPDF

from rag_model_wrap import load_rag_state, rag_answer, client

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="NutriMate AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌙 NutriMate — AI Recipe & Meal Planning Assistant")

# ==========================================================
# LOAD RAG MODEL (ONCE)
# ==========================================================
@st.cache_resource
def init_model():
    return load_rag_state()

recipes, embeddings, embedder, index = init_model()

# ==========================================================
# SESSION STATE
# ==========================================================
if "saved_recipes" not in st.session_state:
    st.session_state.saved_recipes = []

if "last_meal_plan" not in st.session_state:
    st.session_state.last_meal_plan = None

if "last_generated_recipes" not in st.session_state:
    st.session_state.last_generated_recipes = None

if "allergy_list" not in st.session_state:
    st.session_state.allergy_list = []

# store per-meal details so we don't re-call LLM on every rerun
if "meal_details" not in st.session_state:
    # key: meal string, value: dict(name, ingredients, steps, ai_summary)
    st.session_state.meal_details = {}


# ==========================================================
# SMALL HELPERS
# ==========================================================
def safe_latin1(text: str) -> str:
    """Convert text to latin-1-safe string (drop unsupported chars)."""
    return str(text).encode("latin-1", "ignore").decode("latin-1")


# ==========================================================
# ICS BUILDER (SIMPLE)
# ==========================================================
def build_ics(meal_plan_json):
    if not meal_plan_json:
        return b"BEGIN:VCALENDAR\nEND:VCALENDAR"

    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//NutriMate//MealPlan//EN"]
    today = datetime.now()

    for i, d in enumerate(meal_plan_json):
        date = today + timedelta(days=i)
        dtstamp = date.strftime("%Y%m%dT090000")

        lines += [
            "BEGIN:VEVENT",
            f"UID:{i}@nutrimate",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART;VALUE=DATE:{date.strftime('%Y%m%d')}",
            f"SUMMARY:{safe_latin1(d['main_dish'])}",
            "DESCRIPTION:" + safe_latin1("\\n".join(d["meals"])),
            "END:VEVENT"
        ]

    lines.append("END:VCALENDAR")
    return "\n".join(lines).encode("latin-1", "ignore")


# ==========================================================
# JSON-SAFE COMPLETION FOR MEAL PLAN
# ==========================================================
JSON_SYSTEM = """
You MUST output ONLY valid JSON.
No comments. No extra text.
"""

def safe_json_completion(prompt: str):
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": JSON_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.25,
    )
    txt = resp.choices[0].message.content

    try:
        return json.loads(txt)
    except Exception:
        # one repair attempt
        repair = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": JSON_SYSTEM},
                {"role": "user", "content": "Fix this into valid JSON only:\n" + txt},
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        try:
            return json.loads(repair.choices[0].message.content)
        except Exception:
            return []


# ==========================================================
# STRUCTURED PDF BUILDER (USES PRECOMPUTED MEAL DETAILS)
# ==========================================================
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def build_structured_pdf(meal_plan, meal_details):
    buffer = st.session_state.get("pdf_buffer", None)
    from io import BytesIO
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("<b>NutriMate Weekly Meal Plan</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    for day in meal_plan:
        day_title = f"<b>{day['day']} — {day['calories']} kcal</b>"
        story.append(Paragraph(day_title, styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        md = f"<b>Main dish:</b> {day['main_dish']}"
        story.append(Paragraph(md, styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        for meal in day["meals"]:
            story.append(Paragraph(f"<b>{meal}</b>", styles["Heading3"]))
            details = meal_details.get(meal, {})

            # Ingredients
            ingredients = details.get("ingredients", [])
            ing_html = "<br/>".join([f"• {i}" for i in ingredients])
            story.append(Paragraph(f"<b>Ingredients:</b><br/>{ing_html}", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

            # Steps
            steps = details.get("steps", [])
            step_html = "<br/>".join([f"- {s}" for s in steps])
            story.append(Paragraph(f"<b>Steps:</b><br/>{step_html}", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

            # AI Summary
            ai = details.get("ai_summary", "")
            if ai:
                ai_html = ai.replace("\n", "<br/>")
                story.append(Paragraph(f"<b>AI Recommendation:</b><br/>{ai_html}", styles["Normal"]))

            story.append(Spacer(1, 0.2 * inch))

        # Section divider
        divider = Paragraph("<br/><br/>----------------------------------------------<br/><br/>", styles["Normal"])
        story.append(divider)

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data




# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(
    ["🍽 Recipe Generator", "📅 Meal Planner", "⭐ Saved Recipes"]
)

# ==========================================================
# 🍽 RECIPE GENERATOR
# ==========================================================
with tab1:
    st.header("🍽 AI Recipe Generator")

    recipe_prompt = st.text_area(
        "Describe your request:",
        placeholder=(
            "- Chinese dish under 600 calories\n"
            "- Quick chicken dinner in 20 minutes\n"
            "- Gluten-free meal with no peanuts"
        ),
        height=130,
    )

    allergies = st.text_input(
        "Allergies (comma-separated):",
        placeholder="peanuts, dairy, shrimp",
    )
    allergy_list = [a.strip().lower() for a in allergies.split(",") if a.strip()]
    st.session_state.allergy_list = allergy_list

    colA, colB = st.columns(2)
    with colA:
        num_recipes = st.slider("Number of recipes:", 1, 8, 3)
    with colB:
        cal_limit = st.slider("Max calories per recipe:", 200, 1500, 700)

    generate_btn = st.button("Generate Recipes")

    # ---------- generation ----------
    if generate_btn:
        final_prompt = f"{recipe_prompt}\nMax calories per dish: {cal_limit}."

        with st.spinner("Generating recipes..."):
            rag_res = rag_answer(final_prompt, top_k=num_recipes)

            filtered = []
            for rec in rag_res["retrieved"]:
                # STRICT allergy filter before AI
                if any(
                    allg in ing
                    for allg in allergy_list
                    for ing in rec["ingredients"]
                ):
                    continue

                # AI recommendation for this recipe (precomputed)
                analysis_prompt = f"""
You are evaluating ONE recipe against user constraints.

User query: {recipe_prompt}
Declared allergies: {allergy_list}
Recipe name: {rec['name']}
Ingredients: {rec['ingredients']}
Approx calories: {rec['calories']}
Tags: {rec['tags']}

Rules:
1. ONLY consider the DECLARED allergies above. If none of those allergy words appear in the ingredient list, allergy risk must be "no".
2. Do NOT invent new allergies.
3. Consider calories and cuisine/time hints from the user query when judging constraint match. If user did not mention any, do not evaluate. 

Return markdown EXACTLY in this form:

- Allergy risk: yes/no — very short explanation
- Constraint match: short assessment
- Notes: one short sentence
"""
                analysis = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    max_tokens=260,
                    temperature=0.1,
                )
                rec["ai_summary"] = analysis.choices[0].message.content

                filtered.append(rec)

            st.session_state.last_generated_recipes = filtered

    # ---------- display ----------
    placeholder = st.empty()
    with placeholder.container():
        recs = st.session_state.last_generated_recipes
        if recs:
            cols = st.columns(3)
            for i, rec in enumerate(recs):
                col = cols[i % 3]
                with col:
                    st.markdown(f"### 🍲 {rec['name']}")
                    st.markdown(f"**Calories:** {rec['calories']} kcal")
                    st.markdown(f"**Time:** {rec['minutes']} min")

                    st.markdown(
                        " ".join([f"`{t}`" for t in rec["tags"][:6]])
                    )

                    st.markdown("**Ingredients:**")
                    st.write(", ".join(rec["ingredients"][:10]))

                    with st.expander("AI Recommendation"):
                        st.markdown(rec.get("ai_summary", ""))

                    with st.expander("Steps"):
                        for step in rec["steps"]:
                            st.write("- " + step)

                    if st.button("⭐ Save Recipe", key=f"save_{i}"):
                        st.session_state.saved_recipes.append(rec)
                        st.success("Saved!")


# ==========================================================
# 📅 MEAL PLANNER
# ==========================================================
with tab2:
    st.header("📅 Weekly Meal Planner")

    plan_prompt = st.text_area(
        "Dietary goals:",
        placeholder="e.g., 1800 kcal/day, high protein, Asian cuisine, no dairy",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Plan length:", 3, 14, 7)
    with col2:
        daily_limit = st.slider("Daily calorie limit:", 1000, 3000, 1800)

    gen_plan_btn = st.button("Generate Meal Plan")

    # ---------- generate plan & precompute meal details ----------
    if gen_plan_btn:
        today = datetime.now().strftime("%Y-%m-%d")
        json_prompt = f"""
Create a {days}-day meal plan starting {today}.
Daily calorie limit: {daily_limit} kcal.
User constraints: {plan_prompt}

Return ONLY JSON like:
[
  {{
    "day": "Day 1",
    "main_dish": "Beef Stir Fry",
    "calories": 1750,
    "meals": ["Breakfast: ...", "Lunch: ...", "Dinner: ..."]
  }}
]
"""
        with st.spinner("Planning meals..."):
            plan = safe_json_completion(json_prompt)

            # precompute details for each meal once
            meal_details = st.session_state.meal_details
            for day in plan:
                for meal in day["meals"]:
                    if meal in meal_details:
                        continue

                    # get ingredients & steps via RAG
                    detail = rag_answer(
                        f"Give me ingredients and steps for: {meal}",
                        top_k=1,
                    )
                    if len(detail["retrieved"]) == 0:
                        meal_details[meal] = {
                            "name": meal,
                            "ingredients": [],
                            "steps": [],
                            "ai_summary": "",
                        }
                        continue

                    recipe = detail["retrieved"][0]
                    ingredients = recipe["ingredients"]
                    steps = recipe["steps"]

                    # AI summary for this dish in context of goals
                    ai_prompt = f"""
You are evaluating ONE dish within a multi-day plan.

Meal label: {meal}
Ingredients: {ingredients}
User plan description: {plan_prompt}
Daily calorie limit: {daily_limit} kcal

Rules:
1. Allergy risk: ONLY consider the declared allergies. If none of those words appear in the ingredients list, allergy risk must be "no".
2. Constraint match: judge roughly if this meal fits the plan (cuisine, calories, allergic restrictions, health goals).
3. Be concise.

Return markdown exactly:

- Constraint match: short assessment
- Notes: one short sentence
"""
                    ai_resp = client.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[{"role": "user", "content": ai_prompt}],
                        max_tokens=260,
                        temperature=0.1,
                    )
                    ai_summary = ai_resp.choices[0].message.content

                    meal_details[meal] = {
                        "name": meal,
                        "ingredients": ingredients,
                        "steps": steps,
                        "ai_summary": ai_summary,
                    }

            st.session_state.meal_details = meal_details
            st.session_state.last_meal_plan = plan

    # ---------- display plan ----------
    if st.session_state.last_meal_plan:
        plan = st.session_state.last_meal_plan
        meal_details = st.session_state.meal_details

        ph2 = st.empty()
        with ph2.container():
            st.subheader("📆 Weekly Calendar")
            cols = st.columns(4)

            for i, d in enumerate(plan):
                col = cols[i % 4]
                with col:
                    st.markdown(f"### 📅 {d['day']}")
                    st.markdown(f"**Calories:** {d['calories']} kcal")
                    st.markdown(f"**Main dish:** {d['main_dish']}")

                    for meal in d["meals"]:
                        simplified_title = meal.split(" - ")[0].strip()

                        with st.expander(f"📘 {simplified_title}"):
                            md = meal_details.get(meal)
                            if not md:
                                st.warning("No details found.")
                                continue

                            st.markdown("#### Ingredients")
                            st.write(", ".join(md["ingredients"]))

                            st.markdown("#### AI Recommendation")
                            st.markdown(md.get("ai_summary", ""))

                            st.markdown("#### Steps")
                            for step in md["steps"]:
                                st.write("- " + step)


        # ---------- export (no regeneration) ----------
        st.subheader("📤 Export")

        pdf_data = build_structured_pdf(plan, st.session_state.meal_details)
        st.download_button(
            "📄 Download PDF",
            data=pdf_data,
            file_name="meal_plan.pdf",
            mime="application/pdf",
        )

        ics_data = build_ics(plan)
        st.download_button(
            "📅 Add to Calendar",
            data=ics_data,
            file_name="meal_plan.ics",
            mime="text/calendar",
        )


# ==========================================================
# ⭐ SAVED RECIPES
# ==========================================================
with tab3:
    st.header("⭐ Saved Recipes")

    saved = st.session_state.saved_recipes
    if not saved:
        st.info("No saved recipes yet.")
    else:
        for rec in saved:
            with st.expander(f"🍽 {rec['name']}"):
                st.write(f"**Calories:** {rec['calories']}")
                st.write(f"**Time:** {rec['minutes']} min")
                st.write("**Ingredients:**")
                st.write(", ".join(rec["ingredients"]))
                st.write("**Steps:**")
                for s in rec["steps"]:
                    st.write("- " + s)
