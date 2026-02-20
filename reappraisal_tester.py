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
  "conversational_response": "Brief acknowledgement of the user's last point",
  "next_question": "The string of the next CORE QUESTION to ask",
  "final_narrative": "The full synthesized 1st-person narrative (only if status is complete)"
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
    # Clean JSON output from potential markdown formatting
    text = re.sub(r"```json|```", "", res.content).strip()
    return json.loads(text)

# --- 4. APP PAGES ---

def show_consent():
    st.header("📄 Research Participation Consent")
    
    st.markdown("<hr style='margin: 0x 0 5px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
    
    st.write("""
    **Introduction and Purpose** You are invited to participate in a research study exploring how individuals perceive and attribute motives to personal life events.
    
    **Data Privacy and Confidentiality** Your privacy is our priority. All data collected, including your Prolific ID, text narratives, and motive ratings, will be stored securely in an encrypted database.
    * **Anonymization:** Your personal identity is never directly linked to your responses in any public-facing report or publication.
    * **Storage:** Data is used strictly for academic research purposes to improve our understanding of human cognition.

    **Voluntary Participation and Risks** Participation in this study is strictly voluntary.
    * **Right to Withdraw:** You may choose to stop the study at any time by closing the browser window without penalty; however, completion of all sections is required for compensation via Prolific.
    * **Potential Risks:** There are no known physical risks. Some participants may experience mild emotional discomfort when describing a negative life event. You are encouraged to describe events that you feel comfortable sharing in a research context.

    **Consent Statement** By clicking **"Agree and Start Study"** below, you indicate that you are at least 18 years of age, have read and understood the information provided above, and voluntarily agree to participate in this study.
    """)
    
    st.markdown("<hr style='margin: 0x 0 5px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)

    # Prolific ID logic remains untouched
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
    st.markdown(f"Please respond to the assistant below regarding your **{val.lower()}** event.")

    if f"msgs_{idx}" not in st.session_state:
        # Start with the first core question as mandated by the prompt
        init_q = INTERVIEW_QUESTIONS[0] 
        st.session_state.update({f"msgs_{idx}": [AIMessage(content=init_q)], f"hist_{idx}": [], f"curr_q_{idx}": init_q})

    for m in st.session_state[f"msgs_{idx}"]:
        with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"): st.write(m.content)

    if user_input := st.chat_input("Type your response..."):
        st.session_state[f"msgs_{idx}"].append(HumanMessage(content=user_input))
        st.session_state[f"hist_{idx}"].append({"question": st.session_state[f"curr_q_{idx}"], "answer": user_input})
        
        with st.spinner("Processing..."):
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
    st.markdown("Please review the narrative below and edit it to ensure it captures your experience in your own words.")
    
    narrative = st.text_area("Event Narrative:", value=st.session_state[f"raw_narrative_{idx}"], height=250)
    
    st.divider()
    st.subheader("Experience Feedback")
    col1, col2 = st.columns(2)
    h_val = col1.slider("AI Helpfulness (1-9)", 1, 9, 5, key=f"h_{idx}")
    f_val = col2.slider("Conversation Naturalness (1-9)", 1, 9, 5, key=f"f_{idx}")
    fb_text = st.text_area("Additional feedback on the chat interface:", key=f"fb_{idx}")

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
    
    st.header(f"📊 Motive Ratings")
    
    st.markdown("Now, please review your narratives and rate the motives for both events below.")
    
    RADIO_OPTIONS = list(range(1, 10))
    all_scores = {}

    # We use a single form for both events to save data all at once
    with st.form("master_motive_form"):
        
        # Loop through both events (Positive and Negative)
        for idx in [0, 1]:
            val = st.session_state.event_order[idx]
            narrative = st.session_state[f"final_narrative_{idx}"]
            
            st.markdown(f"### {val} Event")
            
            # Narrative in a collapsed expander to keep the view compact
            with st.expander(f"View {val.lower()} event narrative", expanded=False):
                st.write(narrative)
            
            event_scores = {}
            for name, pro, prev in MOTIVES_GOALS:
                col1, col2 = st.columns(2)
                with col1:
                    event_scores[f"{name}_Promotion"] = st.radio(
                        f"{pro}", 
                        options=RADIO_OPTIONS, 
                        index=None, 
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
                # Compact horizontal rule
                st.markdown("<hr style='margin: -8px 0 2px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
            
            all_scores[idx] = event_scores
            
        error_placeholder = st.empty()
        
        # Validation Logic using specific motive descriptions
        if st.form_submit_button("Submit All and Finish Study", type="primary"):
            missing_descriptions = []
            
            for idx in [0, 1]:
                val = st.session_state.event_order[idx]
                for name, pro, prev in MOTIVES_GOALS:
                    # FIX: Match the key structure used in event_scores exactly
                    if all_scores[idx][f"{name}_Promotion"] is None:
                        missing_descriptions.append(f"{val} Event: '{pro}'")
                    if all_scores[idx][f"{name}_Prevention"] is None:
                        missing_descriptions.append(f"{val} Event: '{prev}'")

            if missing_descriptions:
                with error_placeholder.container():
                    st.error(f"⚠️ Some ratings are missing. Please ensure every single row has a selection before continuing.")
                    
                    st.markdown("Please provide a rating for the following specific items:")
                    for desc in missing_descriptions:
                        st.write(f"- {desc}")
            else:
                error_placeholder.empty()
                
                # Save and proceed
                st.session_state["motive_scores_0"] = all_scores[0]
                st.session_state["motive_scores_1"] = all_scores[1]
                
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
    db.collection("prolific_study").add(data)

def show_finish():
    st.balloons()
    st.header("✅ Study Complete")
    st.success("Your responses have been saved.")
    st.link_button("Return to Prolific", f"https://app.prolific.com/submissions/complete?cc={PROLIFIC_COMPLETION_CODE}")

# --- 5. MAIN ROUTER ---
pages = {"consent": show_consent, "chat": show_chat, "review": show_review, "motives": show_motives, "finish": show_finish}
pages[st.session_state.page]()
