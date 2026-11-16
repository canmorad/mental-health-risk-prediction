from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "best_logistic_regression.pkl"
)

PINK, YELLOW, TEAL, CREAM, NAVY = "#C8697B", "#FFCF4D", "#2F6B87", "#FFF8EE", "#1F2A44"

RISK_STYLE = {
    "Low": ("#2F6B87", "fa-circle-check"),
    "Medium": ("#B8860B", "fa-triangle-exclamation"),
    "High": ("#B83A4B", "fa-circle-exclamation"),
}

RISK_TIPS = {
    "Low": "The profile shows mostly protective patterns. Keep up regular sleep, activity and social contact.",
    "Medium": "Some factors may deserve attention. Small changes in sleep, workload or screen time can help, and talking with peers or a student advisor is a good idea.",
    "High": "Several factors point to a higher level of difficulty. Consider reaching out to your university's student support or counselling service. This is not a diagnosis.",
}

st.set_page_config(
    page_title="Care | Mental Health Risk", page_icon="💗", layout="wide"
)

CSS = f"""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Jost:wght@400;500;600&display=swap');

html, body, .stApp {{ font-family:'Jost', sans-serif; color-scheme: light !important; }}
.stApp {{ background:{PINK}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:100% !important; padding:2rem 4rem 3rem !important; }}
@media (max-width:800px) {{ .block-container {{ padding:1.5rem 1rem !important; }} }}

.nav {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem; }}
.logo {{ font-size:1.7rem; font-weight:600; color:#fff; }}
.logo i {{ color:{YELLOW}; margin-right:.4rem; }}
.pill {{ background:{YELLOW}; color:#111; font-weight:600; padding:.5rem 1.2rem;
         border-radius:10px; box-shadow:4px 4px 0 #111; }}
.pill i {{ margin-right:.4rem; }}
.hero {{ display:flex; align-items:center; gap:2rem; flex-wrap:wrap; }}
.hero .txt {{ flex:1.3; min-width:300px; }}
.hero .art {{ flex:1; min-width:260px; text-align:center; }}
.hero .art svg {{ width:100%; max-width:420px; height:auto; }}
.hero h1 {{ font-family:'Playfair Display',serif; font-size:3rem; color:#fff !important; margin:0; line-height:1.15; }}
.hero h1 span {{ color:{YELLOW} !important; }}
.hero .outline {{ font-family:'Playfair Display',serif; font-size:4.6rem; font-weight:800;
                  color:transparent; -webkit-text-stroke:2px {YELLOW}; line-height:1.1; }}
.hero p {{ color:#ffe9ee; font-size:1.1rem; max-width:560px; margin:.8rem 0 1.2rem; }}
.torn {{ height:18px; background:{CREAM}; margin:1.5rem 0;
  clip-path:polygon(0 60%,5% 30%,12% 70%,20% 35%,30% 75%,40% 30%,50% 70%,60% 35%,70% 75%,80% 30%,90% 70%,100% 40%,100% 100%,0 100%); }}

.steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1.2rem; margin-bottom:2rem; }}
.step {{ background:{CREAM}; border-radius:16px; padding:1.2rem 1.4rem; box-shadow:6px 6px 0 {TEAL}; color:{NAVY}; }}
.step i {{ font-size:1.6rem; color:{PINK}; margin-bottom:.5rem; }}
.step b {{ display:block; font-family:'Playfair Display',serif; font-size:1.15rem; margin-bottom:.2rem; }}
.step span {{ font-size:.95rem; }}

[data-testid="stForm"] {{ background:{CREAM}; border:none; border-radius:16px; padding:1.5rem 2rem;
                          box-shadow:6px 6px 0 {TEAL}; }}
[data-testid="stForm"] label p, [data-testid="stForm"] p {{ color:{NAVY} !important; font-weight:500; }}
.sec {{ font-family:'Playfair Display',serif; color:{NAVY}; font-size:1.25rem; font-weight:700;
        margin:.8rem 0 .2rem; border-bottom:2px dashed {PINK}; padding-bottom:.2rem; }}
.sec i {{ color:{PINK}; margin-right:.5rem; }}

[data-testid="stAppViewContainer"] [data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stForm"] [data-baseweb="select"] > div,
[data-baseweb="select"] > div {{
    background-color: #ffffff !important;
    border: 2px solid {NAVY} !important;
    border-top-color: {NAVY} !important;
    border-right-color: {NAVY} !important;
    border-bottom-color: {NAVY} !important;
    border-left-color: {NAVY} !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}
[data-testid="stForm"] [data-baseweb="select"] > div:hover,
[data-testid="stForm"] [data-baseweb="select"] > div:focus-within {{
    border-color: {NAVY} !important;
    box-shadow: 0 0 0 3px rgba(200, 105, 123, 0.25) !important;
}}
[data-baseweb="select"] *, [data-baseweb="select"] input {{ color:{NAVY} !important; -webkit-text-fill-color:{NAVY} !important; }}
[data-baseweb="select"] svg {{ fill:{NAVY} !important; }}
div[data-baseweb="popover"] {{ color-scheme: light !important; }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [data-baseweb="menu"] {{ background:#fff !important; }}
div[data-baseweb="popover"] li {{ background:#fff !important; color:{NAVY} !important; }}
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {{ background:#FFF1C2 !important; }}
div[data-baseweb="popover"] li * {{ color:{NAVY} !important; }}
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color:{NAVY} !important; }}
[data-baseweb="slider"] div[role="slider"] {{ background:{PINK} !important; box-shadow:none !important; border:3px solid {NAVY} !important; }}
[data-baseweb="slider"] div[data-testid="stTickBar"] {{ background:#eadfd0 !important; }}

[data-testid="stFormSubmitButton"] > button {{
    background:{YELLOW}; color:#111; font-weight:600; border:none; border-radius:10px;
    padding:.6rem 1.8rem; box-shadow:4px 4px 0 #111; transition:.1s;
    -webkit-appearance: none !important; appearance: none !important; }}
[data-testid="stFormSubmitButton"] > button:hover {{
    background:{YELLOW}; color:#111; transform:translate(2px,2px); box-shadow:2px 2px 0 #111; }}

h2 {{ font-family:'Playfair Display',serif !important; color:#fff !important; }}
h2 i {{ color:{YELLOW}; margin-right:.5rem; }}

.result {{ background:{CREAM}; border-radius:16px; padding:1.3rem 1.8rem; margin-top:1.2rem;
           box-shadow:6px 6px 0 {TEAL}; color:{NAVY}; }}
.result .lvl {{ font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; }}
.result .lvl i {{ margin-right:.6rem; }}
.result .tip {{ margin-top:.6rem; padding:.8rem 1rem; background:#FFF1C2; border-radius:10px; }}
.result .tip i {{ color:{PINK}; margin-right:.5rem; }}
.bar {{ background:#eadfd0; border-radius:8px; height:14px; overflow:hidden; margin:.2rem 0 .8rem; }}
.bar > div {{ height:100%; }}
.note {{ color:#ffe9ee; font-size:.9rem; margin-top:2rem; }}
.note i {{ color:{YELLOW}; margin-right:.4rem; }}
</style>
"""

HERO = f"""
<div class="nav">
<div class="logo"><i class="fa-solid fa-heart-pulse"></i>Care</div>
<div class="pill"><i class="fa-solid fa-graduation-cap"></i>Educational project</div>
</div>
<div class="hero">
<div class="txt">
<h1>Student well-being, <span>our top priority</span></h1>
<div class="outline">Prevention</div>
<p>A machine-learning prototype that estimates a student's mental-health risk level
(Low / Medium / High) from lifestyle, academic and personal factors.</p>
</div>
<div class="art">
<svg viewBox="0 0 400 320" xmlns="http://www.w3.org/2000/svg">
<path d="M0 230 Q60 200 120 225 T250 215 T400 230 L400 320 L0 320Z" fill="{TEAL}"/>
<path d="M0 245 Q70 220 140 240 T270 232 T400 246" stroke="{CREAM}" stroke-width="10" fill="none" stroke-linecap="round"/>
<circle cx="200" cy="140" r="115" fill="#D98396"/>
<path d="M200 250 C90 170 110 80 170 80 C190 80 200 95 200 108 C200 95 210 80 230 80 C290 80 310 170 200 250Z" fill="{CREAM}"/>
<polyline points="118,160 168,160 184,125 206,200 222,152 282,152" fill="none" stroke="{YELLOW}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="70" cy="70" r="9" fill="{YELLOW}"/><circle cx="335" cy="55" r="6" fill="{CREAM}"/>
<circle cx="350" cy="160" r="8" fill="{YELLOW}"/><circle cx="55" cy="170" r="5" fill="{CREAM}"/>
<path d="M310 95 l6 14 l14 6 l-14 6 l-6 14 l-6 -14 l-14 -6 l14 -6z" fill="{YELLOW}"/>
</svg>
</div>
</div>
<div class="torn"></div>
<div class="steps">
<div class="step"><i class="fa-solid fa-pen-to-square"></i><b>1. Fill in the profile</b><span>Describe sleep, study habits, screen time and personal context.</span></div>
<div class="step"><i class="fa-solid fa-brain"></i><b>2. The model analyses it</b><span>A saved pipeline (preprocessing + logistic regression) handles everything.</span></div>
<div class="step"><i class="fa-solid fa-lightbulb"></i><b>3. Read the insight</b><span>Get the predicted level, class probabilities and general wellbeing advice.</span></div>
</div>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)


def section(icon: str, title: str):
    st.markdown(
        f'<div class="sec"><i class="fa-solid {icon}"></i>{title}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

classes = list(getattr(model, "classes_", ["Low", "Medium", "High"]))
EXPECTED = list(getattr(model, "feature_names_in_", []))

if "Student_ID" in EXPECTED:
    st.error(
        "The saved model still expects Student_ID. "
        "Please retrain the model without Student_ID."
    )
    st.stop()

st.markdown(
    '<h2><i class="fa-solid fa-user-pen"></i>Student information</h2>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):

    section("fa-id-card", "Profile")
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 15, 60, 21)
    gender = c2.selectbox("Gender", ["Female", "Male", "Non-binary"])
    year = c3.slider("Year of study", 1, 8, 2)

    section("fa-moon", "Lifestyle")
    c1, c2, c3 = st.columns(3)
    sleep = c1.slider("Sleep (hours / night)", 0.0, 14.0, 6.8, 0.1)
    activity = c2.selectbox("Physical activity", ["Low", "Moderate", "High"], index=1)
    diet = c3.selectbox(
        "Diet quality", ["Poor", "Average", "Good", "Not specified"], index=1
    )

    section("fa-book-open", "Academic life")
    c1, c2, c3 = st.columns(3)
    study = c1.slider("Study hours / week", 0.0, 80.0, 20.0, 0.5)
    gpa = c2.slider("GPA (0-4)", 0.0, 4.0, 2.6, 0.01)
    attendance = c3.slider("Attendance rate (%)", 0.0, 100.0, 85.0, 0.5)

    section("fa-mobile-screen", "Screen use")
    c1, c2 = st.columns(2)
    social = c1.slider("Social media (hours / day)", 0.0, 24.0, 6.0, 0.1)
    screen = c2.slider("Total screen time (hours / day)", 0.0, 24.0, 8.5, 0.1)

    section("fa-hand-holding-heart", "Personal context")
    c1, c2, c3, c4 = st.columns(4)
    fin = c1.slider("Financial stress (0-10)", 0.0, 10.0, 6.0, 0.1)
    fam = c2.slider("Family support (0-10)", 0.0, 10.0, 6.5, 0.1)
    job = c3.selectbox("Part-time job", ["No", "Yes"])
    hist = c4.selectbox("Previous mental-health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict risk level")


if submitted:

    if social > screen:
        st.warning(
            "Social media time is higher than total screen time. Please check these two values."
        )

    try:
        X = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Year_of_Study": year,
                    "Sleep_Hours": sleep,
                    "Study_Hours_Per_Week": study,
                    "GPA": gpa,
                    "Social_Media_Hours": social,
                    "Screen_Time_Hours": screen,
                    "Physical_Activity": activity,
                    "Financial_Stress": fin,
                    "Family_Support": fam,
                    "Attendance_Rate": attendance,
                    "Part_Time_Job": job,
                    "Diet_Quality": np.nan if diet == "Not specified" else diet,
                    "Previous_Mental_Health_History": hist,
                }
            ]
        )

        for col in EXPECTED:
            if col not in X.columns:
                X[col] = np.nan

        if EXPECTED:
            X = X[EXPECTED]

        pred = str(model.predict(X)[0])

        color, icon = RISK_STYLE.get(pred, (NAVY, "fa-circle-info"))

        html = (
            f'<div class="result"><div class="lvl" style="color:{color}">'
            f'<i class="fa-solid {icon}"></i>Predicted risk level: {pred}</div><br>'
        )

        if hasattr(model, "predict_proba"):
            for c, p in zip(classes, model.predict_proba(X)[0]):
                bc = RISK_STYLE.get(str(c), (PINK,))[0]

                html += (
                    f"<div>{c}: <b>{p:.1%}</b></div>"
                    f'<div class="bar"><div style="width:{p*100:.0f}%;background:{bc}"></div></div>'
                )

        if pred in RISK_TIPS:
            html += (
                f'<div class="tip"><i class="fa-solid fa-hand-holding-heart"></i>'
                f"{RISK_TIPS[pred]}</div>"
            )

        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"Prediction error: {exc}")


st.markdown(
    '<div class="note"><i class="fa-solid fa-triangle-exclamation"></i>Strictly educational project: '
    "this model is not a diagnostic or clinical screening tool, and must never be used for disciplinary "
    "decisions or student selection.</div>",
    unsafe_allow_html=True,
)