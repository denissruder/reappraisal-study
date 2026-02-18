import streamlit as st
import os
import json
import datetime
import uuid
import random
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 0. CONFIGURATION & UI SETUP ---
MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.8
PROLIFIC_COMPLETION_CODE = "YOUR_CODE_HERE" 

st.set_page_config(page_title="Psychological Study", layout="centered")

# Original CSS Injection for tight UI and column borders
st.markdown("""
<style>
.stForm { max-width: 900px; margin: 0 auto; padding: 5px; }
div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"] { margin: 0 !important; padding: 0 !important; }
h1, h2, h3, h4 { margin-top: 0.5rem !important; margin-bottom: 0.2rem !important; padding-top: 0.25rem !important; }
.stForm h4 { margin-top: 10px !important; margin-bottom: 5px !important; border-bottom: 1px solid #ddd; }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) { border-left: 1px solid #ccc; padding-left: 15px; }
div[role="radiogroup"] { gap: 0px !important; }
div[role="radiogroup"] label { font-size: 0.9rem !important; margin-right: 5px !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. CORE DATA ---
MOTIVES_GOALS = [
    ("Hedonic", "To feel good", "Not to feel bad"), ("Physical", "To be in good health", "To stay safe"),
    ("Wealth", "To have money", "To avoid poverty"), ("Predictability", "To understand", "To avoid confusion"),
    ("Competence", "To succeed", "To avoid failure"), ("Growth", "To learn and grow", "To avoid monotony or decline"),
    ("Autonomy", "To be free to decide", "Not to be told what to do"), ("Relatedness", "To feel connected", "To avoid loneliness"),
    ("Acceptance", "To be liked", "To avoid disapproval"), ("Status", "To stand out", "To avoid being ignored"),
    ("Responsibility", "To live up to expectations", "Not to let others down"), ("Meaning", "To make a difference", "Not to waste my life"),
    ("Instrumental", "To gain something", "To avoid something"),
]

INTERVIEW_QUESTIONS = ["What happened?", "Why?", "Is it finished?", "How big a deal?", "Desired outcome?", "Responsibility?", "Changeability?", "Expectations?"]

# --- 2. LLM & DB SETUP ---
@st.cache_resource
def get_llm():
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)

@st.cache_resource
def get_db():
    from google.cloud import firestore
    key_dict = json.loads(st.secrets["gcp_service_account"], strict=False)
    return firestore.Client.from_service_account_info(key_dict)

llm = get_llm()
db = get_db()

# --- 3. LOGIC ---
def process_interview_step(history, valence):
    qa_pairs = "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history])
    prompt = f"""
    Analyze the user's description of a {valence} event. 
    History: {qa_pairs}
    Return JSON: {{"status": "continue"|"complete", "conversational_response": "...", "next_question": "...", "final_narrative": "..."}}
    """
    res = llm.invoke(prompt)
    text = re.sub(r"```json|```", "", res.content).strip()
    return json.loads(text)

# --- 4. APP PAGES ---
def show_consent():
    st.title("📄 Consent Form")
    pid = st.query_params.get("PROLIFIC_PID", f"test_{uuid.uuid4().hex[:6]}")
    st.session_state.prolific_id = pid
    if st.button("I Consent", type="primary"):
        orders = ["Positive", "Negative"]
        random.shuffle(orders)
        st.session_state.update({"event_order": orders, "current_idx": 0, "page": "chat", "results": {}})
        st.rerun()

def show_chat():
    idx = st.session_state.current_idx
    val = st.session_state.event_order[idx]
    st.header(f"Step {idx+1}/2: Describe a {val} Event")

    if f"msgs_{idx}" not in st.session_state:
        init_q = f"Could you describe a recent **{val.lower()}** emotionally significant event?"
        st.session_state[f"msgs_{idx}"] = [AIMessage(content=init_q)]
        st.session_state[f"hist_{idx}"] = []
        st.session_state[f"curr_q_{idx}"] = init_q

    for m in st.session_state[f"msgs_{idx}"]:
        with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"): st.write(m.content)

    if user_input := st.chat_input("Your response..."):
        st.session_state[f"msgs_{idx}"].append(HumanMessage(content=user_input))
        st.session_state[f"hist_{idx}"].append({"question": st.session_state[f"curr_q_{idx}"], "answer": user_input})
        
        res = process_interview_step(st.session_state[f"hist_{idx}"], val)
        if res['status'] == 'complete':
            st.session_state[f"raw_narrative_{idx}"] = res['final_narrative']
            st.session_state.page = "review"
        else:
            msg = f"{res['conversational_response']} {res['next_question']}"
            st.session_state[f"msgs_{idx}"].append(AIMessage(content=msg))
            st.session_state[f"curr_q_{idx}"] = res['next_question']
        st.rerun()

def show_review():
    idx = st.session_state.current_idx
    st.header("📝 Review & Experience")
    narrative = st.text_area("Edit your story:", value=st.session_state[f"raw_narrative_{idx}"], height=250)
    
    col1, col2 = st.columns(2)
    h_val = col1.slider("Helpfulness (1-9)", 1, 9, 5, key=f"h_{idx}")
    f_val = col2.slider("Naturalness (1-9)", 1, 9, 5, key=f"f_{idx}")
    fb_text = st.text_area("Comments on AI interaction?", key=f"fb_{idx}")

    if st.button("Confirm Event Description"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        st.session_state[f"feedback_{idx}"] = {"help": h_val, "flow": f_val, "text": fb_text}
        
        if idx == 0:
            st.session_state.current_idx = 1
            st.session_state.page = "chat"
        else:
            st.session_state.current_idx = 0 # Reset to 0 for the Motive Rating order
            st.session_state.page = "motives"
        st.rerun()

def show_motives():
    idx = st.session_state.current_idx
    val = st.session_state.event_order[idx]
    st.header(f"📊 Motive Appraisal: {val} Event ({idx+1}/2)")
    st.info(st.session_state[f"final_narrative_{idx}"])

    scores = {}
    with st.form(f"motive_form_{idx}"):
        for name, pro, prev in MOTIVES_GOALS:
            st.markdown(f"#### {name}")
            c1, c2 = st.columns(2)
            scores[f"{name}_Promotion"] = c1.radio(pro, range(1,10), index=4, horizontal=True)
            scores[f"{name}_Prevention"] = c2.radio(prev, range(1,10), index=4, horizontal=True)

        if st.form_submit_button("Submit Ratings"):
            st.session_state[f"motive_scores_{idx}"] = scores
            if idx == 0:
                st.session_state.current_idx = 1
            else:
                save_data()
                st.session_state.page = "finish"
            st.rerun()

def save_data():
    data = {
        "prolific_id": st.session_state.prolific_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "event_order": st.session_state.event_order,
        "payload": []
    }
    for i in range(2):
        data["payload"].append({
            "valence": st.session_state.event_order[i],
            "full_chat_history": [{"q": x['question'], "a": x['answer']} for x in st.session_state[f"hist_{i}"]],
            "initial_synthesized_narrative": st.session_state[f"raw_narrative_{i}"],
            "user_edited_narrative": st.session_state[f"final_narrative_{i}"],
            "user_experience": st.session_state[f"feedback_{i}"],
            "motive_ratings": st.session_state[f"motive_scores_{i}"]
        })
    db.collection("prolific_study_v2").add(data)

def show_finish():
    st.balloons()
    st.title("✅ Completed")
    st.link_button("Return to Prolific", f"https://app.prolific.com/submissions/complete?cc={PROLIFIC_COMPLETION_CODE}")

# --- 5. ROUTER ---
pages = {"consent": show_consent, "chat": show_chat, "review": show_review, "motives": show_motives, "finish": show_finish}
pages[st.session_state.page]()
