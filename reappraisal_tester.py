import streamlit as st
import os
import json
import datetime
import uuid
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 0. CONFIGURATION ---
# Use "gemini-1.5-flash" or "gemini-1.5-pro" as these are the standard stable names.
MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.8
PROLIFIC_COMPLETION_CODE = "YOUR_CODE_HERE" # Replace with your actual code

st.set_page_config(page_title="Psychological Study", layout="centered")

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

INTERVIEW_QUESTIONS = [
    "What happened?", "As far as you can tell, why did things happen the way they did?",
    "Is this situation finished or not?", "How big of a deal is this for you?",
    "What would you have wanted to happen instead?", "Who is responsible?",
    "Could you still change this?", "Did things go as expected?"
]

# --- 2. LLM & DB SETUP ---
@st.cache_resource
def get_llm():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("API Key missing.")
        st.stop()
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP, google_api_key=st.secrets["GEMINI_API_KEY"])

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
        res = llm.invoke(prompt)
        return {"status": "complete", "final_narrative": res.content.strip()}

    prompt = PromptTemplate.from_template("""
    Analyze this interview history for a {valence} event.
    History: {qa_pairs}
    
    If the event is well-described, set status 'complete' and write a 'final_narrative' (1st person).
    Otherwise, set status 'continue' and provide 'next_question'.
    
    Return ONLY JSON:
    {{
      "status": "continue/complete",
      "conversational_response": "...",
      "next_question": "...",
      "final_narrative": "..."
    }}
    """)
    
    try:
        res = llm.invoke(prompt.format(valence=valence, qa_pairs=qa_pairs))
        # Remove any markdown code blocks from the response
        clean_json = res.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"status": "error", "conversational_response": "I'm having trouble. Can you tell me more?", "next_question": "What else happened?"}

# --- 4. APP PAGES ---

def show_consent():
    st.title("📄 Participant Consent")
    pid = st.query_params.get("PROLIFIC_PID", f"test_{uuid.uuid4().hex[:6]}")
    st.session_state.prolific_id = pid
    
    st.write(f"Welcome, Participant {pid}. You will describe two recent life events.")
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
    
    # Progress Bar: 0.1 to 0.5 for first event, 0.6 to 1.0 for second
    progress_val = (idx * 0.5) + 0.2
    st.progress(progress_val, text=f"Event {idx+1} of 2 ({valence})")

    if f"history_{idx}" not in st.session_state:
        st.session_state[f"history_{idx}"] = []
        st.session_state[f"next_q_{idx}"] = INTERVIEW_QUESTIONS[0]

    history = st.session_state[f"history_{idx}"]
    
    for item in history:
        with st.chat_message("user"): st.write(item['answer'])
        with st.chat_message("assistant"): st.write(item['response'])

    if user_input := st.chat_input("Tell me more..."):
        with st.spinner("Processing..."):
            res = process_interview_step(history + [{"question": st.session_state[f"next_q_{idx}"], "answer": user_input}], valence)
            
            history.append({
                "question": st.session_state[f"next_q_{idx}"],
                "answer": user_input,
                "response": res.get("conversational_response", "")
            })
            
            if res.get('status') == 'complete':
                st.session_state[f"narrative_{idx}"] = res['final_narrative']
                st.session_state.page = "review"
            else:
                st.session_state[f"next_q_{idx}"] = res.get('next_question', "Can you elaborate?")
            st.rerun()

def show_review():
    idx = st.session_state.current_event_idx
    st.header("📝 Review Narrative")
    
    narrative = st.text_area("Final Narrative (1st Person):", value=st.session_state[f"narrative_{idx}"], height=200)
    
    st.subheader("How was this AI interaction?")
    rating = st.select_slider("Helpfulness (1=Poor, 9=Excellent)", options=range(1,10), value=5)
    feedback = st.text_area("Any comments on the AI interviewer?")

    if st.button("Confirm Narrative"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        st.session_state[f"feedback_{idx}"] = {"rating": rating, "text": feedback}
        st.session_state.page = "motives"
        st.rerun()

def show_motives():
    idx = st.session_state.current_event_idx
    st.header(f"🧐 Motive Importance: {st.session_state.event_order[idx]} Event")
    
    st.info(st.session_state[f"final_narrative_{idx}"])
    
    scores = {}
    with st.form(f"motive_form_{idx}"):
        for name, pro, prev in MOTIVES_GOALS:
            st.markdown(f"**{name}**")
            c1, c2 = st.columns(2)
            scores[f"{name}_Promotion"] = c1.radio(f"{pro}", range(1,10), index=4, horizontal=True)
            scores[f"{name}_Prevention"] = c2.radio(f"{prev}", range(1,10), index=4, horizontal=True)
            st.divider()
        
        if st.form_submit_button("Submit Ratings"):
            st.session_state[f"motive_scores_{idx}"] = scores
            if idx == 0:
                st.session_state.current_event_idx = 1
                st.session_state.page = "chat"
            else:
                save_and_finish()
            st.rerun()

def save_and_finish():
    data = {
        "prolific_id": st.session_state.prolific_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "event_order": st.session_state.event_order,
        "data": [
            {
                "valence": st.session_state.event_order[i],
                "narrative": st.session_state[f"final_narrative_{i}"],
                "feedback": st.session_state[f"feedback_{i}"],
                "motives": st.session_state[f"motive_scores_{i}"]
            } for i in range(2)
        ]
    }
    db.collection("prolific_study").add(data)
    st.session_state.page = "finish"

def show_finish():
    st.balloons()
    st.title("✅ Completed")
    st.write("Thank you! Click the button below to return to Prolific and finish.")
    completion_url = f"https://app.prolific.com/submissions/complete?cc={PROLIFIC_COMPLETION_CODE}"
    st.link_button("Return to Prolific", completion_url, type="primary")

# --- 5. ROUTING ---
pages = {
    "consent": show_consent, "chat": show_chat, "review": show_review, 
    "motives": show_motives, "finish": show_finish
}
pages[st.session_state.page]()
