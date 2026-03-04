import streamlit as st
import os
import json
import datetime
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CORE CONFIG & LOGIC LOADING ---
import toml
from reapprasal-study_llm_config import run_interviewer_turn, run_synthesizer, config

# Retrieve API Key from Secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize Database
from google.cloud import firestore
@st.cache_resource
def get_db():
    key_dict = json.loads(st.secrets["gcp_service_account"], strict=False)
    return firestore.Client.from_service_account_info(key_dict)

db = get_db()

# --- 2. UI SETUP ---
st.set_page_config(page_title=config["interface"]["browser_title"], layout="centered")

# Load Motives from TOML
MOTIVES_DATA = config["motives"]["dimensions"]

if 'page' not in st.session_state:
    st.session_state.page = 'consent'

# --- 3. HELPER FUNCTIONS ---

def init_event_state(idx):
    """Initializes tracking for a specific event (0 or 1)."""
    if f"scores_{idx}" not in st.session_state:
        # State Vector initialized to 0
        st.session_state[f"scores_{idx}"] = {k: 0 for k in config["chat"]["questions"].keys()}
    if f"msgs_{idx}" not in st.session_state:
        # Start with the first question from TOML
        st.session_state[f"msgs_{idx}"] = [AIMessage(content=config["chat"]["questions"]["event"])]
    if f"hist_{idx}" not in st.session_state:
        st.session_state[f"hist_{idx}"] = []
    if f"curr_q_{idx}" not in st.session_state:
        st.session_state[f"curr_q_{idx}"] = config["chat"]["questions"]["event"]

def is_saturated(scores):
    """Python-side gate to decide if the interview is done."""
    # Required: Event and Feeling must be high (>= 6)
    if scores.get("event", 0) < 6 or scores.get("feeling", 0) < 6:
        return False
    # Breadth: 5/8 dimensions must be sufficiently covered
    covered = [s for s in scores.values() if s >= 6]
    return len(covered) >= 5

def save_to_firestore():
    """Bundles all session data, scores, and AI reasoning into Firestore."""
    data = {
        "prolific_id": st.session_state.prolific_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "event_results": []
    }
    
    for i in range(2): # For both events
        data["event_results"].append({
            "valence": st.session_state.event_order[i],
            "chat_history": st.session_state[f"hist_{i}"],
            "saturation_scores": st.session_state[f"scores_{i}"], # Saving the state vector
            "ai_reasoning": st.session_state.get(f"reasoning_{i}"), # Saving the CoT/CoV log
            "final_narrative": st.session_state[f"final_narrative_{i}"],
            "motive_ratings": st.session_state[f"motive_scores_{i}"],
            "ux_metrics": st.session_state.get(f"ux_metrics_{i}")
        })
    
    db.collection("prolific_study_results").add(data)

# --- 4. PAGE RENDERING ---

def show_consent():
    st.header(config["consent"]["header"])
    st.markdown(config["consent"]["body"])
    
    st.session_state.prolific_id = st.query_params.get("PROLIFIC_PID", "test_user")
    
    if st.button(config["consent"]["confirm_button"], type="primary"):
        st.session_state.event_order = ["Negative", "Positive"]
        st.session_state.current_idx = 0
        st.session_state.page = "chat"
        st.rerun()

def show_chat():
    idx = st.session_state.current_idx
    val = st.session_state.event_order[idx]
    init_event_state(idx)
    
    st.header(config["chat"]["header"].format(val=val))
    st.info(config["chat"]["body"])

    # Render History
    for m in st.session_state[f"msgs_{idx}"]:
        with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
            st.write(m.content)

    if user_input := st.chat_input(config["chat"]["input_placeholder"]):
        st.session_state[f"msgs_{idx}"].append(HumanMessage(content=user_input))
        st.session_state[f"hist_{idx}"].append({
            "question": st.session_state[f"curr_q_{idx}"], 
            "answer": user_input
        })
        
        with st.spinner(config["interface"]["loading_state"]):
            # 1. Interviewer Turn
            res = run_interviewer_turn(GEMINI_API_KEY, st.session_state[f"hist_{idx}"], st.session_state[f"scores_{idx}"])
            
            # 2. Update Scores
            st.session_state[f"scores_{idx}"] = res.coverage_scores
            
            # 3. Decision Gate
            if is_saturated(res.coverage_scores):
                # 4. Synthesis
                synth = run_synthesizer(GEMINI_API_KEY, st.session_state[f"hist_{idx}"])
                st.session_state[f"raw_narrative_{idx}"] = synth.final_narrative
                st.session_state[f"reasoning_{idx}"] = synth.reasoning # Store reasoning log
                st.session_state.page = "review"
            else:
                st.session_state[f"msgs_{idx}"].append(AIMessage(content=res.conversational_response))
                st.session_state[f"curr_q_{idx}"] = res.conversational_response
        st.rerun()

def show_review():
    idx = st.session_state.current_idx
    st.header(config["review"]["header"])
    st.markdown(config["review"]["body"])
    
    narrative = st.text_area("Final Narrative:", value=st.session_state[f"raw_narrative_{idx}"], height=300)
    
    st.subheader("Feedback")
    f1 = st.slider(config["review"]["rating_1"], 1, 9, 5)
    f2 = st.slider(config["review"]["rating_2"], 1, 9, 5)

    if st.button(config["review"]["confirm_button"], type="primary"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        st.session_state[f"ux_metrics_{idx}"] = {"captures": f1, "voice": f2}
        
        # Progression logic
        if idx == 0:
            st.session_state.current_idx = 1
            st.session_state.page = "chat"
        else:
            st.session_state.current_idx = 0 
            st.session_state.page = "motives"
        st.rerun()

def show_motives():
    idx = st.session_state.current_idx
    st.header(config["motives"]["header"])
    st.markdown(config["motives"]["body"])
    
    motive_scores = {}
    with st.form(f"motive_form_{idx}"):
        st.info(st.session_state[f"final_narrative_{idx}"])
        for dim in MOTIVES_DATA:
            c1, c2 = st.columns(2)
            motive_scores[f"{dim['name']}_Promotion"] = c1.radio(dim["promotion"], range(1, 10), index=4, horizontal=True)
            motive_scores[f"{dim['name']}_Prevention"] = c2.radio(dim["prevention"], range(1, 10), index=4, horizontal=True)
            st.markdown("---")
            
        if st.form_submit_button(config["motives"]["submit_button"]):
            st.session_state[f"motive_scores_{idx}"] = motive_scores
            if idx == 0:
                st.session_state.current_idx = 1
                st.rerun()
            else:
                with st.spinner("Finalizing study..."):
                    save_to_firestore()
                st.session_state.page = "finish"
                st.rerun()

# --- 5. ROUTER ---
pages = {
    "consent": show_consent, "chat": show_chat, 
    "review": show_review, "motives": show_motives,
    "finish": lambda: st.header(config["exit"]["header"])
}
pages[st.session_state.page]()
