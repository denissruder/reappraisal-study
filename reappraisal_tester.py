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

# --- 0. CONFIGURATION & UI ---
MODEL_NAME = "gemini-1.5-flash" # Updated to current stable
TEMP = 0.8
RATING_SCALE_MAX = 9
MIN_NARRATIVE_LENGTH = 100

st.set_page_config(page_title="Psychological Study", layout="centered")

# CSS for a clean research interface
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .motive-box { padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px; }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
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
    "What happened?",
    "As far as you can tell, why did things happen the way they did?",
    "Is this situation finished or not? If not, what could happen next?",
    "How big of a deal is this situation for you? Why?",
    "What would you have wanted to happen in this situation instead of what actually happened?",
    "Who do you feel is responsible for how this situation unfolded?",
    "Could you still change this situation if you wanted to? How?",
    "Did things go as you expected? If not, what was unexpected?"
]

# --- 2. LLM & DB SETUP ---
@st.cache_resource
def get_llm():
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)

@st.cache_resource
def get_db():
    from google.cloud import firestore
    key_dict = json.loads(st.secrets["gcp_service_account"], strict=False)
    return firestore.Client.from_service_account_info(key_dict)

llm = get_llm()
db = get_db()

# --- 3. LOGIC FUNCTIONS ---

def process_interview_step(history, valence, is_skip=False):
    qa_pairs = "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history])
    
    if is_skip:
        prompt = f"Synthesize these answers into a first-person narrative about a {valence} event: {qa_pairs}"
        response = llm.invoke(prompt)
        return {"status": "complete", "final_narrative": response.content.strip()}

    prompt_template = PromptTemplate.from_template("""
    Role: Dynamic Interviewer for a {valence} event.
    History: {qa_pairs}
    All Questions: {all_questions}
    
    Task: If the story is complete, set status 'complete'. Otherwise 'continue' and ask the next relevant question.
    Output JSON: {{"status": "...", "conversational_response": "...", "next_question": "...", "final_narrative": "..."}}
    """)
    
    chain = prompt_template | llm
    res = chain.invoke({"valence": valence, "qa_pairs": qa_pairs, "all_questions": "\n".join(INTERVIEW_QUESTIONS)})
    # Simple cleaner for JSON response
    cleaned = res.content.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)

# --- 4. APP PAGES ---

def show_consent():
    st.title("📄 Study Consent")
    st.write("Welcome. This study involves describing a recent positive and a recent negative event.")
    
    # Capture Prolific PID from URL
    prolific_id = st.query_params.get("PROLIFIC_PID", "Unknown")
    st.session_state.prolific_id = prolific_id

    if st.button("I Consent", type="primary"):
        # Randomize order
        orders = ["Positive", "Negative"]
        random.shuffle(orders)
        st.session_state.event_order = orders
        st.session_state.current_event_idx = 0
        st.session_state.page = "chat"
        st.rerun()

def show_chat():
    idx = st.session_state.current_event_idx
    valence = st.session_state.event_order[idx]
    
    st.header(f"Step {idx+1}/2: {valence} Event Interview")
    st.info(f"Please tell me about a recent **{valence.lower()}** emotionally significant event.")

    if f"chat_history_{idx}" not in st.session_state:
        st.session_state[f"chat_history_{idx}"] = []
        st.session_state[f"next_q_{idx}"] = INTERVIEW_QUESTIONS[0]

    history = st.session_state[f"chat_history_{idx}"]
    
    for item in history:
        with st.chat_message("user"): st.write(item['answer'])
        with st.chat_message("assistant"): st.write(item['response'])

    if user_input := st.chat_input("Describe the event..."):
        with st.spinner("Thinking..."):
            res = process_interview_step(history + [{"question": st.session_state[f"next_q_{idx}"], "answer": user_input}], valence)
            
            history.append({
                "question": st.session_state[f"next_q_{idx}"],
                "answer": user_input,
                "response": res.get("conversational_response", "")
            })
            
            if res['status'] == 'complete':
                st.session_state[f"narrative_{idx}"] = res['final_narrative']
                st.session_state.page = "review"
            else:
                st.session_state[f"next_q_{idx}"] = res['next_question']
            st.rerun()

    if len(history) > 1:
        if st.button("Finish & Synthesize Now"):
            res = process_interview_step(history, valence, is_skip=True)
            st.session_state[f"narrative_{idx}"] = res['final_narrative']
            st.session_state.page = "review"
            st.rerun()

def show_review():
    idx = st.session_state.current_event_idx
    st.header("📝 Review & Experience Rating")
    
    narrative = st.text_area("Edit your story for accuracy:", value=st.session_state[f"narrative_{idx}"], height=250)
    
    st.subheader("How was your experience during this interview?")
    col1, col2 = st.columns(2)
    with col1:
        rating = st.slider("How helpful was the AI in helping you describe the event?", 1, 10, 5)
    with col2:
        flow = st.slider("How natural was the conversation?", 1, 10, 5)
        
    feedback = st.text_area("Any additional thoughts on the AI interaction? (Optional)")

    if st.button("Confirm & Proceed"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        st.session_state[f"feedback_{idx}"] = {"rating": rating, "flow": flow, "text": feedback}
        st.session_state.page = "motives"
        st.rerun()

def show_motives():
    idx = st.session_state.current_event_idx
    valence = st.session_state.event_order[idx]
    
    st.header("🧐 Motive Importance")
    with st.expander("Show My Confirmed Event Narrative", expanded=True):
        st.write(st.session_state[f"final_narrative_{idx}"])

    st.write("Rate how important each motive was **during this specific event**.")
    
    scores = {}
    with st.form(f"motive_form_{idx}"):
        for name, pro_text, prev_text in MOTIVES_GOALS:
            st.markdown(f"### {name}")
            c1, c2 = st.columns(2)
            scores[f"{name}_Promotion"] = c1.radio(f"{pro_text}", range(1,10), index=4, horizontal=True)
            scores[f"{name}_Prevention"] = c2.radio(f"{prev_text}", range(1,10), index=4, horizontal=True)
            st.divider()
        
        if st.form_submit_button("Submit Event Data"):
            st.session_state[f"motive_scores_{idx}"] = scores
            if st.session_state.current_event_idx == 0:
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
        "event_order": st.session_state.event_order,
        "events": []
    }
    for i in range(2):
        data["events"].append({
            "valence": st.session_state.event_order[i],
            "narrative": st.session_state[f"final_narrative_{i}"],
            "feedback": st.session_state[f"feedback_{i}"],
            "motives": st.session_state[f"motive_scores_{i}"]
        })
    db.collection("prolific_study").add(data)

# --- 5. MAIN ROUTING ---
if "page" not in st.session_state: st.session_state.page = "consent"

pages = {
    "consent": show_consent,
    "chat": show_chat,
    "review": show_review,
    "motives": show_motives,
    "finish": lambda: st.success("Thank you! Your participation is complete. You may now return to Prolific.")
}

pages[st.session_state.page]()
