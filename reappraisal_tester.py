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

# --- 0. CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.8
PROLIFIC_COMPLETION_CODE = "C1234567" # TODO: Replace with your actual code
MIN_NARRATIVE_LENGTH = 100

st.set_page_config(page_title="Psychological Study", layout="wide")

# Inject your original CSS for the clean look
st.markdown("""
<style>
.stForm { max-width: 900px; margin: 0 auto; padding: 5px; }
div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"] { margin: 0 !important; padding: 0 !important; }
h1, h2, h3, h4 { margin-top: 0.5rem !important; margin-bottom: 0.2rem !important; }
div[role="radiogroup"] { gap: 0px !important; }
div[role="radiogroup"] label { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. CORE DATA ---
MOTIVES_GOALS = [
    ("Hedonic", "To feel good", "Not to feel bad"),
    ("Physical", "To be in good health", "To stay safe"),
    ("Wealth", "To have money", "To avoid poverty"),
    ("Predictability", "To understand", "To avoid confusion"),
    ("Competence", "To succeed", "To avoid failure"),
    ("Growth", "To learn and grow", "To avoid monotony or decline"),
    ("Autonomy", "To be free to decide", "Not to be told what to do"),
    ("Relatedness", "To feel connected", "To avoid loneliness"),
    ("Acceptance", "To be liked", "To avoid disapproval"),
    ("Status", "To stand out", "To avoid being ignored"),
    ("Responsibility", "To live up to expectations", "Not to let others down"),
    ("Meaning", "To make a difference", "Not to waste my life"),
    ("Instrumental", "To gain something", "To avoid something"),
]

INTERVIEW_QUESTIONS = [
    "What happened?", "Why did things happen that way?", "Is it finished?",
    "How big of a deal is this?", "What did you want to happen?",
    "Who is responsible?", "Could you change it?", "Was it expected?"
]

# --- 2. LLM & DB SETUP ---
@st.cache_resource
def get_llm():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("API Key missing in secrets.")
        st.stop()
    # Set both to be safe
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)

@st.cache_resource
def get_db():
    from google.cloud import firestore
    key_dict = json.loads(st.secrets["gcp_service_account"], strict=False)
    return firestore.Client.from_service_account_info(key_dict)

llm = get_llm()
db = get_db()

# --- 3. DYNAMIC PROMPTS ---

INTERVIEW_PROMPT = """
# ROLE: Dynamic Interviewer for Psychological Study
Analyze the user's description of a {valence} event.
History: {qa_pairs}
Core Questions: {all_questions}

Task:
1. Determine if the description is rich and complete (status: complete) or needs more info (status: continue).
2. Write a conversational response and select the next relevant question if continuing.
3. If complete, synthesize a first-person narrative (using 'I' and 'my').

Output MUST be valid JSON:
{{
  "status": "continue" | "complete",
  "conversational_response": "...",
  "next_question": "...",
  "final_narrative": "..."
}}
"""

# --- 4. APP LOGIC ---

def process_interview_step(history, valence):
    qa_pairs = "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history])
    prompt = INTERVIEW_PROMPT.format(
        valence=valence, 
        qa_pairs=qa_pairs, 
        all_questions="\n".join(INTERVIEW_QUESTIONS)
    )
    
    try:
        res = llm.invoke(prompt)
        text = res.content.strip()
        # Clean markdown if present
        if text.startswith("```json"): text = text[7:-3]
        elif text.startswith("```"): text = text[3:-3]
        return json.loads(text)
    except Exception as e:
        return {"status": "error", "conversational_response": "I see. Tell me more.", "next_question": "What else happened?"}

def show_consent():
    st.title("📄 Consent Form")
    pid = st.query_params.get("PROLIFIC_PID", f"test_{uuid.uuid4().hex[:6]}")
    st.session_state.prolific_id = pid
    
    st.write(f"Welcome Participant **{pid}**. This study involves two brief interviews about life events.")
    if st.button("I Consent", type="primary"):
        orders = ["Positive", "Negative"]
        random.shuffle(orders)
        st.session_state.event_order = orders
        st.session_state.current_event_idx = 0
        st.session_state.page = "chat"
        st.rerun()

def show_chat():
    idx = st.session_state.current_event_idx
    valence = st.session_state.event_order[idx]
    
    st.header(f"Event {idx+1}/2: {valence} Experience")
    
    # Init state for this specific event
    if f"chat_msgs_{idx}" not in st.session_state:
        initial_q = f"Please describe a recent **{valence.lower()}** emotionally significant event."
        st.session_state[f"chat_msgs_{idx}"] = [AIMessage(content=initial_q)]
        st.session_state[f"hist_{idx}"] = []
        st.session_state[f"curr_q_{idx}"] = initial_q

    # Display chat
    for m in st.session_state[f"chat_msgs_{idx}"]:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role): st.write(m.content)

    if user_input := st.chat_input("Your response..."):
        st.session_state[f"chat_msgs_{idx}"].append(HumanMessage(content=user_input))
        st.session_state[f"hist_{idx}"].append({
            "question": st.session_state[f"curr_q_{idx}"], 
            "answer": user_input
        })
        
        with st.spinner("AI is thinking..."):
            res = process_interview_step(st.session_state[f"hist_{idx}"], valence)
            
            if res['status'] == 'complete':
                st.session_state[f"narrative_{idx}"] = res['final_narrative']
                st.session_state.page = "review"
            else:
                resp = f"{res.get('conversational_response', '')} {res.get('next_question', '')}"
                st.session_state[f"chat_msgs_{idx}"].append(AIMessage(content=resp))
                st.session_state[f"curr_q_{idx}"] = res.get('next_question')
            st.rerun()

def show_review():
    idx = st.session_state.current_event_idx
    st.header("📝 Narrative Review & Feedback")
    
    narrative = st.text_area("Edit your story for accuracy:", value=st.session_state[f"narrative_{idx}"], height=250)
    
    st.subheader("How was the AI interviewer?")
    c1, c2 = st.columns(2)
    help_val = c1.slider("Helpfulness (1-9)", 1, 9, 5)
    flow_val = c2.slider("Naturalness (1-9)", 1, 9, 5)
    feedback = st.text_area("Any additional comments on the AI interaction?")

    if st.button("Confirm Narrative"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        st.session_state[f"feedback_{idx}"] = {"help": help_val, "flow": flow_val, "text": feedback}
        st.session_state.page = "motives"
        st.rerun()

def show_motives():
    idx = st.session_state.current_event_idx
    st.header(f"📊 Motive Appraisal: {st.session_state.event_order[idx]} Event")
    
    st.info(st.session_state[f"final_narrative_{idx}"])
    
    scores = {}
    with st.form(f"motive_form_{idx}"):
        for name, pro, prev in MOTIVES_GOALS:
            st.markdown(f"**{name}**")
            c1, c2 = st.columns(2)
            scores[f"{name}_Promotion"] = c1.radio(f"{pro}", range(1,10), index=4, horizontal=True)
            scores[f"{name}_Prevention"] = c2.radio(f"{prev}", range(1,10), index=4, horizontal=True)
            st.divider()
        
        if st.form_submit_button("Submit Event Data"):
            st.session_state[f"scores_{idx}"] = scores
            if idx == 0:
                st.session_state.current_event_idx = 1
                st.session_state.page = "chat"
            else:
                save_final_data()
                st.session_state.page = "finish"
            st.rerun()

def save_final_data():
    data = {
        "prolific_id": st.session_state.prolific_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "events": [
            {
                "valence": st.session_state.event_order[i],
                "narrative": st.session_state[f"final_narrative_{i}"],
                "feedback": st.session_state[f"feedback_{i}"],
                "motives": st.session_state[f"scores_{i}"]
            } for i in range(2)
        ]
    }
    db.collection("prolific_study").add(data)

def show_finish():
    st.balloons()
    st.title("✅ Study Complete")
    st.success("Your data has been saved.")
    url = f"[https://app.prolific.com/submissions/complete?cc=](https://app.prolific.com/submissions/complete?cc=){PROLIFIC_COMPLETION_CODE}"
    st.link_button("Return to Prolific to complete", url, type="primary")

# --- 5. MAIN ROUTER ---
if "page" not in st.session_state: st.session_state.page = "consent"

pages = {
    "consent": show_consent, "chat": show_chat, 
    "review": show_review, "motives": show_motives, "finish": show_finish
}
pages[st.session_state.page]()
