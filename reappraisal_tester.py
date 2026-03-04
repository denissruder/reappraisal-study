import streamlit as st
import os
import json
import datetime
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Inject minimal CSS for a cleaner, tighter look
st.markdown("""
<style>

/* 0. NEW: Page-wide top margin reduction */
.block-container {
    padding-top: 2rem !important;
}

/* 1. Global Container/Form Spacing Reduction */
.stForm {
    max-width: 900px;
    margin: 0 auto;
    padding: 5px; 
}
/* Aggressively zero out vertical space for all internal blocks */
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    margin: 0 !important;
    padding: 0 !important;
}

/* 2. Header Spacing Reduction */
h1, h2, h3, h4 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.2rem !important; 
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
}

/* 3. Style for motive headers (H4) inside the form */
.stForm h4 {
    margin-top: 10px !important; 
    margin-bottom: 5px !important; 
    padding-bottom: 3px !important;
    border-bottom: 1px solid #ddd;
}

/* 4. Column Border Fix */
/* CRITICAL: Target the second column in the row to apply a left border. 
   Streamlit columns are internally represented as horizontal blocks.
   We select the second horizontal block (column) of the current row.
*/
div[data-testid="stHorizontalBlock"]:nth-child(2) {
    border-left: 1px solid #ccc; /* Light grey vertical border */
    padding-left: 10px; /* Add padding to push content away from the border */
}
/* Ensure the first column also has padding for balance */
div[data-testid="stHorizontalBlock"]:nth-child(1) {
    padding-right: 10px; 
}

/* 5. Radio Button Spacing Fixes */
div[data-testid^="stRadio"] {
    margin-bottom: -20px !important; 
    margin-top: -10px !important;
}
div[role="radiogroup"] {
    gap: 0px !important; 
}
div[role="radiogroup"] label {
    margin-right: 5px !important; 
    padding: 0px !important;
}

/* 6. Submit Button Spacing */
div[data-testid="stFormSubmitButton"] {
    padding-top: 5px; 
    padding-bottom: 5px;
}
div[role="radiogroup"] label {
    /* ... existing styles ... */
    font-size: 0.9rem !important; /* NEW: Makes the radio button labels slightly smaller */
}

/* Disable all transitions and animations for instant switching */
div[data-testid="stNotification"], 
div[data-testid="stForm"], 
div[data-testid="stVerticalBlock"] {
    animation: none !important;
    transition: none !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# --- 1. CORE CONFIG & LOGIC LOADING ---
import toml
from reapprasal_study_llm_config import run_interviewer_turn, run_synthesizer, config

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
    
    # 1. Narrative Review Area
    narrative = st.text_area("Edit your story:", value=st.session_state[f"raw_narrative_{idx}"], height=300)
    
    # 2. Qualitative Feedback Ratings (All 4 items from TOML)
    st.divider()
    st.subheader("How accurate is this summary?")
    
    r1 = st.slider(config["review"]["rating_1"], 1, 9, 5)
    r2 = st.slider(config["review"]["rating_2"], 1, 9, 5)
    r3 = st.slider(config["review"]["rating_3"], 1, 9, 5)
    r4 = st.slider(config["review"]["rating_4"], 1, 9, 5)
    
    # 3. Open-ended Feedback
    user_feedback = st.text_area(
        config["review"]["feedback"], 
        height=100,  # This ensures it is roughly 3-4 lines tall
        placeholder="Type your thoughts here..."
    )  

    if st.button(config["review"]["confirm_button"], type="primary"):
        st.session_state[f"final_narrative_{idx}"] = narrative
        
        # Package all 5 data points for the research log
        st.session_state[f"ux_metrics_{idx}"] = {
            "captures_experience": r1,
            "sounds_like_me": r2,
            "left_out_aspects": r3,
            "confabulated_details": r4,
            "qualitative_notes": user_feedback
        }
        
        # Progression logic to next event or final motive phase
        if idx == 0:
            st.session_state.current_idx = 1
            st.session_state.page = "chat"
        else:
            st.session_state.current_idx = 0 
            st.session_state.page = "motives"
        st.rerun()
        
def show_motives():
    # Use the persistent session state index to determine which event to show
    idx = st.session_state.current_idx
    val = st.session_state.event_order[idx]
    narrative = st.session_state[f"final_narrative_{idx}"]
    
    # Use Header from TOML
    st.header(config["motives"]["header"])
    # Use Body from TOML
    st.markdown(config["motives"]["body"])
    
    RADIO_OPTIONS = list(range(1, 10))
    event_scores = {}

    with st.form(f"motive_form_{idx}"):
        # Narrative reference - Keep as unexpanded expander per your original
        with st.expander(f"View your {val.lower()} event story", expanded=False):
            st.write(narrative)
            
        # Motive rating grid using dimensions from TOML
        for dim in config["motives"]["dimensions"]:
            name = dim["name"]
            pro = dim["promotion"]
            prev = dim["prevention"]
            
            col1, col2 = st.columns(2)
            with col1:
                event_scores[f"{name}_Promotion"] = st.radio(
                    f"{pro}", 
                    options=RADIO_OPTIONS, 
                    index=None, # Mandatory selection
                    horizontal=True, 
                    key=f"sit_{name}_pro_{idx}"
                )
            with col2:
                event_scores[f"{name}_Prevention"] = st.radio(
                    f"{prev}", 
                    options=RADIO_OPTIONS, 
                    index=None, 
                    horizontal=True, 
                    key=f"sit_{name}_prev_{idx}"
                )
            # Original tight border styling
            st.markdown("<hr style='margin: -8px 0 2px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)

    # Use Submit Button text from TOML
   if st.form_submit_button(config["motives"]["submit_button"]):
        # 1. Identify all required dimension names from TOML
        required_names = [dim["name"] for dim in config["motives"]["dimensions"]]
        
        # 2. Find which ones are missing a selection
        missing_promotion = [n for n in required_names if event_scores.get(f"{n}_Promotion") is None]
        missing_prevention = [n for n in required_names if event_scores.get(f"{n}_Prevention") is None]
        
        # Combine unique names that have at least one missing radio selection
        all_missing = sorted(list(set(missing_promotion + missing_prevention)))
        
        if all_missing:
            # 3. Show a specific error message listing the skipped motives
            st.error(f"Please provide ratings for the following motives: {', '.join(all_missing)}")
            return
        
        # If all are filled, save and progress
        st.session_state[f"motive_scores_{idx}"] = event_scores
        
        if idx == 0:
            st.session_state.current_idx = 1
            st.rerun()
        else:
            with st.spinner("Saving your data..."):
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
