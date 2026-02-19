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

if 'page' not in st.session_state:
    st.session_state.page = 'consent'

# Injecting original CSS
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
INTERVIEW_QUESTIONS = [
    "What happened?",
    "How did the situation leave you feeling?",
    "What were you trying to do? What did you want or need in this situation?",
    "In what ways did this situation help or hurt you?",
    "How much did this situation matter to you? Why?",
    "Who or what did you feel was most responsible for this situation?",
    "Did you feel like you were in control in this situation? Why?",
    "Is there anything else that was important to you in this situation?"
]

MOTIVES_GOALS = [
    ("Health", "To be energetic and fit", "To avoid illness and injury"),
    ("Wealth", "To be well off", "To avoid losing out"),
    ("Relatedness", "To be liked and loved", "To avoid rejection and loneliness"),
    ("Status", "To lead and be respected", "To avoid shame and disrespect"),
    ("Purpose", "To serve something beyond myself and make a difference", "To avoid wasting my life or meaningless pursuits"),
    ("Competence", "To get things done", "To avoid mistakes"),
    ("Growth", "To experience and learn", "To avoid boredom and decline"),
    ("Control", "To be free and authentic", "To avoid losing control")
]

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

# --- 3. DYNAMIC INTERVIEWER PROMPT ---
INTERVIEW_PROMPT_TEMPLATE = """ 
#ROLE: Dynamic Interviewer for Psychological Study 
You are an Interviewer for a psychological study. Your goal is to collect all 8 key pieces of information (CORE QUESTIONS) about an event from the user's responses. You may skip asking some of these questions if the user has already responded to them. 

Based on a user’s answers to structured questions, your job is to write the key parts of what they said as a coherent narrative. It is important to capture their experience exactly as they described it. The goal is to produce a set of consistent, raw, first-person stories that reflect the user's experience in their own words — without adding interpretation, analysis, or emotional softening. 

Your responses must be conversational and contextual. You should remain neutral and avoid asking questions in a biased or suggestive way. 

The user's response history so far is: {qa_pairs} 
The set of ALL 8 CORE QUESTIONS is: {all_questions} 

Your task is: 
1. Always begin with the Event and Feelings questions. 
2. Analyze the Q&A history to determine which CORE QUESTIONS have been sufficiently covered by the user's answers. 
3. Not all 8 CORE QUESTIONS must be explicitly covered. Use your best judgment to transition to synthesis when the event description feels rich and complete, or if a remaining question is implicitly answered or clearly non-applicable to the specific event. 
4. If the event description is rich and complete (all necessary points covered), set 'status' to "complete". 
5. If the description is incomplete, set 'status' to "continue". Select the single most relevant and important unanswered question from the list to ask next.

Return your response in JSON format exactly like this:
{{
  "status": "continue" or "complete",
  "conversational_response": "Brief acknowledgement",
  "next_question": "Next CORE QUESTION string",
  "final_narrative": "Full synthesized 1st-person narrative (if complete)"
}}
"""

def process_interview_step(history, valence):
    qa_pairs = "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history])
    formatted_prompt = INTERVIEW_PROMPT_TEMPLATE.format(
        valence=valence,
        qa_pairs=qa_pairs,
        all_questions="\n".join(INTERVIEW_QUESTIONS)
    )
    res = llm.invoke(formatted_prompt)
    text = re.sub(r"```json|```", "", res.content).strip()
    return json.loads(text)

# --- 4. APP PAGES ---

def show_consent():
    st.title("📄 Research Participation Consent")
    st.markdown("Welcome to our study. You will describe one positive and one negative event.")
    pid = st.query_params.get("PROLIFIC_PID", f"test_{uuid.uuid4().hex[:6]}")
    st.session_state.prolific_id = pid
    if st.button("Agree and Start Study", type="primary"):
        orders = ["Positive", "Negative"]
        random.shuffle(orders)
        st.session_state.update({"event_order": orders, "current_idx": 0, "page": "chat"})
        st.rerun()

def show_chat():
    idx = st.session_state.current_idx
    val = st.session_state.event_order[idx]
    
    st.header(f"Phase 1: Describe your {val} Event")
    
    if f"msgs_{idx}" not in st.session_state:
        init_q = INTERVIEW_QUESTIONS[0]
        st.session_state.update({f"msgs_{idx}": [AIMessage(content=init_q)], f"hist_{idx}": [], f"curr_q_{idx}": init_q})

    # Display Chat History
    for m in st.session_state[f"msgs_{idx}"]:
        with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"): 
            st.write(m.content)

    # Use a container to place the spinner exactly where the next message will appear
    spinner_placeholder = st.empty()

    if user_input := st.chat_input("Type your response..."):
        # 1. Update user view immediately
        st.session_state[f"msgs_{idx}"].append(HumanMessage(content=user_input))
        st.session_state[f"hist_{idx}"].append({"question": st.session_state[f"curr_q_{idx}"], "answer": user_input})
        
        # 2. Trigger rerun to show the user's message before AI processing
        st.rerun()

    # If the last message was from the user, we process AI response
    if st.session_state[f"msgs_{idx}"] and isinstance(st.session_state[f"msgs_{idx}"][-1], HumanMessage):
        with spinner_placeholder:
            with st.chat_message("assistant"):
                with st.spinner("Processing your response..."):
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
    val = st.session_state.event_order[idx]
    st.header("📝 Narrative Review")
    narrative = st.text_area("Event Narrative:", value=st.session_state[f"raw_narrative_{idx}"], height=250)
    
    col1, col2 = st.columns(2)
    h_val = col1.slider("AI Helpfulness (1-9)", 1, 9, 5, key=f"h_{idx}")
    f_val = col2.slider("Conversation Naturalness (1-9)", 1, 9, 5, key=f"f_{idx}")
    fb_text = st.text_area("Feedback on chat interface:", key=f"fb_{idx}")

    if st.button("Confirm Narrative"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        st.session_state[f"feedback_{idx}"] = {"help": h_val, "flow": f_val, "text": fb_text}
        
        if idx == 0:
            st.session_state.current_idx = 1
            st.session_state.page = "chat"
        else:
            st.session_state.current_idx = 0 
            st.session_state.page = "motives"
        st.rerun()

def show_motives():
    idx = st.session_state.current_idx
    val = st.session_state.event_order[idx]
    st.header(f"Phase 2: Motive Ratings ({val} Event)")
    
    with st.expander("Reference: Your Narrative", expanded=True):
        st.write(st.session_state[f"final_narrative_{idx}"])

    scores = {}
    with st.form(f"motive_form_{idx}"):
        st.markdown("**1 = Not Important | 9 = Extremely Important**")
        for name, pro, prev in MOTIVES_GOALS:
            st.markdown(f"#### {name}")
            c1, c2 = st.columns(2)
            scores[f"{name}_Promotion"] = c1.radio(f"Goal: {pro}", range(1,10), index=4, horizontal=True)
            scores[f"{name}_Prevention"] = c2.radio(f"Avoidance: {prev}", range(1,10), index=4, horizontal=True)

        if st.form_submit_button("Submit and Continue"):
            st.session_state[f"motive_scores_{idx}"] = scores
            if idx == 0:
                st.session_state.current_idx = 1
            else:
                with st.spinner("Saving data..."):
                    save_to_firestore()
                st.session_state.page = "finish"
            st.rerun()

def save_to_firestore():
    data = {
        "prolific_id": st.session_state.prolific_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "event_order": st.session_state.event_order,
        "results": [
            {
                "valence": st.session_state.event_order[i],
                "chat_history": st.session_state[f"hist_{i}"],
                "initial_narrative": st.session_state[f"raw_narrative_{i}"],
                "final_narrative": st.session_state[f"final_narrative_{i}"],
                "ux_ratings": st.session_state[f"feedback_{i}"],
                "motive_scores": st.session_state[f"motive_scores_{i}"]
            } for i in range(2)
        ]
    }
    db.collection("study_v2_results").add(data)

def show_finish():
    st.balloons()
    st.title("✅ Study Complete")
    st.success("Your responses have been saved.")
    st.link_button("Return to Prolific", f"https://app.prolific.com/submissions/complete?cc={PROLIFIC_COMPLETION_CODE}")

# --- 5. MAIN ROUTER ---
pages = {"consent": show_consent, "chat": show_chat, "review": show_review, "motives": show_motives, "finish": show_finish}
pages[st.session_state.page]()
