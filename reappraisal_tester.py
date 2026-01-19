import streamlit as st
import os
import json
import datetime
import uuid
import time
import random
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from collections import Counter

# --- 0. Streamlit UI Setup ---

st.set_page_config(page_title="Version A: RFT Prediction Study")

# Inject minimal CSS for a cleaner, tighter look
st.markdown("""
<style>

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
    margin-bottom: -5px !important; 
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

# --- 1. CONFIGURATION & SETUP ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.8 # Increased temperature for more diverse CoTs
RATING_SCALE_MIN = 1
RATING_SCALE_MAX = 9 
MIN_NARRATIVE_LENGTH = 100

# CORE DATA: Single source of truth for all motives and their goals
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

# Regulatory Focus Questionnaire items (18 items)
REG_FOCUS_ITEMS = [
    "In general, I am focused on preventing negative events in my life",
    "I am anxious that I will fall short of my responsibilities and obligations",
    "I frequently imagine how I will achieve my hopes and aspirations",
    "I often think about the person I am afraid I might become in the future",
    "I often think about the person I would ideally like to be in the future",
    "I typically focus on the success I hope to achieve in the future",
    "I often worry that I will fail to accomplish my goals",
    "I often think about how I will achieve success",
    "I often imagine myself experiencing bad things that I fear might happen to me",
    "I frequently think about how I can prevent failures in my life",
    "I am more oriented toward preventing losses than I am toward achieving gains",
    "A major goal I have right now is to achieve my ambitions",
    "A major goal I have right now is to avoid becoming a failure",
    "I see myself as someone who is primarily striving to reach my “ideal self” – to fulfill my hopes, wishes, and aspirations",
    "I see myself as someone who is primarily striving to become the self I “ought” to be – to fulfill my duties, responsibilities, and obligations",
    "In general, I am focused on achieving positive outcomes in my life",
    "I often imagine myself experiencing good things that I hope will happen to me",
    "Overall, I am more oriented toward achieving success than preventing failure"
]

# Guided Interview Questions (8 items)
INTERVIEW_QUESTIONS = [
    "What happened? Describe a recent emotionally unpleasant event.",
    "As far as you can tell, why did things happen the way they did?",
    "Is this situation finished or not? If not, what could happen next?",
    "How big of a deal is this situation for you? Why?",
    "What would you have wanted to happen in this situation instead of what actually happened?",
    "Who do you feel is responsible for how this situation unfolded?",
    "Could you still change this situation if you wanted to? How?",
    "Did things go as you expected? If not, what was unexpected?"
]

# Create the master list of 26 required JSON keys
MOTIVE_SCORE_KEYS = []
for motive_name, _, _ in MOTIVES_GOALS:
    MOTIVE_SCORE_KEYS.append(f"{motive_name}_Promotion")
    MOTIVE_SCORE_KEYS.append(f"{motive_name}_Prevention")

JSON_KEYS_LIST = ", ".join(MOTIVE_SCORE_KEYS)

# --- FOR STREAMLIT PAGE LOGIC ---
MOTIVES_FULL = [
    {'motive': m[0], 'Promotion': m[1], 'Prevention': m[2]}
    for m in MOTIVES_GOALS
]

# --- UTIL FUNCTIONS --- 

def parse_llm_json(response_content, attempt_number=0):
    """
    Parses the custom tuple format: Motive_Type : (Score, Justification)
    Example match: Hedonic_Promotion : (1, The individual is...)
    """
    prediction_scores = {}
    reasoning_blocks = []
    
    # Regex breakdown:
    # ([\w]+)                -> Capture Motive name (e.g., Hedonic_Promotion)
    # \s*:\s*\(              -> Match the colon and opening parenthesis
    # (\d+)                  -> Capture the integer score
    # \s*,\s* -> Match the comma separator
    # (.*?)                  -> Capture the justification text (non-greedy)
    # \)                     -> Match the closing parenthesis

    pattern = r"([\w]+)\s*:\s*\(\s*(\d+)\s*,\s*(.*?)\)"
    
    matches = re.findall(pattern, response_content, re.MULTILINE | re.DOTALL)
    
    for motive_name, score, justification in matches:
        # Cast score to int
        prediction_scores[motive_name.strip()] = int(score)
        # Store justification for the 'Reasoning' log
        reasoning_blocks.append(f"{motive_name}: {justification.strip()}")

    # Validation: Ensure we got all 26 motives (13 pairs)
    # Replace JSON_KEYS_LIST with your actual list of 26 keys
    if not prediction_scores:
        return None, "Error: No motives found in the expected format."
        
    reasoning_text = "\n".join(reasoning_blocks)
    return prediction_scores, reasoning_text

def flatten_motive_dict(nested_dict):
    """Converts {'Motive': {'Promotion': X, 'Prevention': Y}} to flat keys."""
    flat = {}
    if not nested_dict: return flat
    for motive, focuses in nested_dict.items():
        if isinstance(focuses, dict):
            for focus_type, score in focuses.items():
                flat[f"{motive}_{focus_type}"] = score
        else:
            flat[motive] = focuses
    return flat

# --- 1. LLM Initialization and Database Setup ---

@st.cache_resource
def get_llm():
    """Initializes the LLM."""
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("LLM Error: 'GEMINI_API_KEY' secret not found. Please check your secrets.")
        st.stop()
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    # Pass TEMP to the LLM constructor
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)
    return llm

llm = get_llm()

try:
    from google.cloud import firestore
except ImportError:
    st.error("The 'google-cloud-firestore' package is required. Please install it (`pip install google-cloud-firestore`).")
    st.stop()

@st.cache_resource
def get_firestore_client():
    """Initializes the Firestore client."""
    if "gcp_service_account" not in st.secrets:
        st.error("Database Error: 'gcp_service_account' secret not found.")
        st.stop()
    try:
        key_dict = json.loads(st.secrets["gcp_service_account"], strict=False)
        db = firestore.Client.from_service_account_info(key_dict)
        return db
    except Exception as e:
        st.error(f"❌ Database Connection Failed: {e}")
        st.stop()
        
db = get_firestore_client()
COLLECTION_NAME = "reappraisal_study_db"
        
# --- LLM INTERVIEW SYNTHESIS (DYNAMIC LOGIC IMPLEMENTATION) ---

# --- Dynamic Interview Logic and Synthesis (Uses first-person 'I') ---
INTERVIEW_PROMPT_TEMPLATE = """
# ROLE: Dynamic Interviewer for Psychological Study
You are a Dynamic Interviewer for a psychological study. Your goal is to collect all 8 key pieces of information (CORE QUESTIONS) about a stressful event from the user's responses, but only ask questions that are relevant or missing.

Your responses must be conversational and contextual.

The user's response history so far is:
{qa_pairs}

The set of ALL 8 CORE QUESTIONS is:
{all_questions}

Your task is:
1. Analyze the Q&A history to determine which CORE QUESTIONS have been sufficiently covered by the user's answers.
2. **CRITICAL RULE:** **Not all 8 CORE QUESTIONS must be explicitly covered.** Use your best judgment to transition to synthesis when the event description feels rich and complete, or if a remaining question is implicitly answered or clearly non-applicable to the specific event.
3. If the event description is rich and complete (all necessary points covered), set 'status' to "complete".
4. If the description is incomplete, set 'status' to "continue". Select the single most relevant and important *unanswered* question from the list to ask next.

Your output MUST be a valid JSON object.

JSON Schema:
{{
  "status": "continue" | "complete",
  "conversational_response": "<A natural, contextual reaction acknowledging the user's last input.>",
  "next_question": "<The full text of the next question to ask, or null if status is 'complete'.>",
  "final_narrative": "<The cohesive, unified story based on ALL answers. This MUST be written from a first-person perspective (using 'I' and 'my'). Only required if status is 'complete'.>"
}}

Provide the JSON output:
"""

# Helper template for manual skip button: synthesize current answers into one narrative.
SYNTHESIS_PROMPT_TEMPLATE = """
# ROLE: Narrative Synthesizer
The user has ended the interview early. Based on the Q&A history provided below, synthesize all the information into a single, cohesive, narrative summary of the stressful event.

**CRITICAL:** Write the summary from a first-person perspective (using 'I' and 'my').

Q&A History:
{qa_pairs}

Provide the complete, unified narrative summary:
"""

def process_interview_step(llm_instance, interview_history, is_skip=False):
    """Executes the LLM to manage the conversation flow or synthesize the narrative."""
    
    qa_pairs_formatted = "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in interview_history])
    
    if is_skip:
        # If user clicked skip, force synthesis immediately
        prompt = PromptTemplate(input_variables=["qa_pairs"], template=SYNTHESIS_PROMPT_TEMPLATE)
        chain = prompt | llm_instance
        try:
            response = chain.invoke({"qa_pairs": qa_pairs_formatted})
            # For manual skip, we generate the narrative directly and wrap it.
            return {
                "status": "complete",
                "conversational_response": "Understood. Thank you for sharing your story so far. I've compiled everything into a single narrative for the next step.",
                "next_question": None,
                "final_narrative": response.content.strip()
            }
        except Exception as e:
            st.error(f"Error during manual Synthesis: {e}")
            return {"status": "error", "conversational_response": "I ran into an issue while compiling your story. Please try submitting the narrative in the next step manually.", "next_question": None, "final_narrative": "ERROR: Could not synthesize story."}


    # Standard dynamic interview flow
    all_questions_formatted = "\n".join([f"- {q}" for q in INTERVIEW_QUESTIONS])
    
    prompt = PromptTemplate(
        input_variables=["qa_pairs", "all_questions"], 
        template=INTERVIEW_PROMPT_TEMPLATE
    )
    chain = prompt | llm_instance
    
    try:
        response = chain.invoke(
            {
                "qa_pairs": qa_pairs_formatted, 
                "all_questions": all_questions_formatted
            }
        )
        json_string = response.content.strip()
        
        # Robust JSON parsing
        if json_string.startswith("```json"):
            json_string = json_string.lstrip("```json").rstrip("```")
        elif json_string.startswith("```"):
            json_string = json_string.lstrip("```").rstrip("```")

        # Attempt simple load
        result = json.loads(json_string, strict=False) 
        return result
    except Exception as e:
        # Fallback error handling
        st.error(f"Error during LLM Interview Processing. Error: {e}")
        # Return a safe, basic structure to continue the flow
        return {"status": "error", "conversational_response": "I ran into an issue while processing that. Can you please tell me more about what happened?", "next_question": INTERVIEW_QUESTIONS[0], "final_narrative": None}


# --- 3. DATA SAVING LOGIC ---

def save_data(data):
    """Saves the trial data to Firestore."""
    try:
        db.collection(COLLECTION_NAME).add(data)
        return True
    except Exception as e:
        st.error(f"❌ Failed to save data: {e}. Check Firestore rules and credentials.")
        return False


# --- 4. STREAMLIT PAGE RENDERING FUNCTIONS (MODIFIED) ---

def show_consent_page():
    """Renders the Letter of Consent page."""
    st.title("📄 Letter of Consent")
    st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #FFF;'>", unsafe_allow_html=True)

    st.markdown("""
    Welcome to the Cognitive Repurposing Study. Before we begin, please review the following information.

    **Purpose:** You will be asked to describe a stressful event and then receive AI-generated guidance designed to help you reframe the situation. The goal is to study how different types of personalization affect the perceived helpfulness of the guidance.

    **Data Collection:** All text input, the guidance you receive, and your final ratings will be stored anonymously in our research database (Firestore). Your identity will not be attached to the data.

    **Risk:** There are no known risks beyond those encountered in daily life. You may stop at any time.

    By clicking 'I Consent,' you agree to participate in this simulated study protocol.
    """)
    
    if st.button("I Consent", type="primary"):
        st.session_state.page = 'regulatory'
        st.rerun()

def show_regulatory_only_page():
    # Define the 1-9 radio options
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'reg_focus_scores' not in st.session_state:
        st.session_state.reg_focus_scores = {item: 5 for item in REG_FOCUS_ITEMS}

    with st.form("regulatory_assessment_form"):
        st.subheader("Regulatory Focus (General Tendency)")
        st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #FFF;'>", unsafe_allow_html=True)
        st.markdown(f"Please indicate how true the following **{len(REG_FOCUS_ITEMS)} statements** are of you **in general** on a scale of 1 to {RATING_SCALE_MAX}.")
        st.markdown(f"**1 = Not At All True of Me** | **{RATING_SCALE_MAX} = Very True of Me**")
        
        reg_focus_scores = st.session_state.reg_focus_scores
        
        # --- Loop through all 18 items sequentially (FIXED) ---
        # enumerate provides the 0-based index (i), which we use for the key, 
        # and the item string, which is used for the dictionary key and display.
        for i, item in enumerate(REG_FOCUS_ITEMS):
            st.markdown("<hr style='margin: 0px 0 10px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
            # The display index is i+1, ensuring continuous numbering from 1 to 18.
            st.session_state.reg_focus_scores[item] = st.radio(
                f"**{i+1}.** {item}", 
                options=RADIO_OPTIONS, 
                # index logic is correct: score minus 1
                index=reg_focus_scores[item] - 1, 
                horizontal=True, 
                key=f"reg_focus_{i}"
            )

        if st.form_submit_button("Next: General Motive Profile", type="primary"):
            st.session_state.page = 'motives' # Route to motives next            
            st.rerun()

def show_motives_only_page():
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'general_motive_scores' not in st.session_state:
        st.session_state.general_motive_scores = {
            m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL
        }

    # Header and Pop-up outside the form
    st.header("📊 General Motive Profile")
    st.info(f"Please rate the importance of the following {len(MOTIVES_FULL)} motives to you in general. **1 = Not Important At All** | **{RATING_SCALE_MAX} = Extremely Important**")

    with st.form("initial_assessment_form"):
        motive_scores = st.session_state.general_motive_scores
        for m in MOTIVES_FULL:
            col1, col2 = st.columns(2) 
            with col1:
                motive_scores[m['motive']]['Promotion'] = st.radio(
                    f"{m['Promotion']}", 
                    options=RADIO_OPTIONS, 
                    index=motive_scores[m['motive']]['Promotion'] - 1, 
                    horizontal=True, 
                    key=f"gen_{m['motive']}_Promotion"
                )
            with col2:
                motive_scores[m['motive']]['Prevention'] = st.radio(
                    f"{m['Prevention']}", 
                    options=RADIO_OPTIONS, 
                    index=motive_scores[m['motive']]['Prevention'] - 1, 
                    horizontal=True, 
                    key=f"gen_{m['motive']}_Prevention"
                )
            st.markdown("<hr style='margin: 0px 0 5px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
            

        if st.form_submit_button("Next: Start Interview", type="primary"):
            st.session_state.page = 'chat' 
            st.rerun()
            
def show_chat_page():
    st.header("🗣️ Event Interview")
    st.markdown("Please describe a recent emotionally unpleasant event. The chatbot will ask follow-up questions to gather necessary context. **You can stop the interview process if your event text is long enough, by clicking the button below.**")

    if 'interview_messages' not in st.session_state:
        initial_question = INTERVIEW_QUESTIONS[0]
        st.session_state.interview_messages = [AIMessage(content=initial_question)]
        st.session_state.interview_answers = []
        st.session_state.next_question = initial_question 
        st.session_state.event_text_synthesized = None

    messages = st.session_state.interview_messages
    answers = st.session_state.interview_answers
    
    # --- NEW: Calculate total character count of answers to enable/disable Skip button ---
    total_chars = sum(len(a['answer']) for a in answers)
    skip_disabled = total_chars < 1000 # Button is greyed out until 1000 chars reached

    chat_container = st.container(height=450, border=True)

    with chat_container:
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
                
    if st.session_state.event_text_synthesized:
        st.session_state.page = 'review_narrative'
        st.rerun()
        
    # Manual skip button with dynamic disabled state
    if st.button(
        f"Skip to Narrative Synthesis ({total_chars}/1000 chars)", 
        type="secondary", 
        use_container_width=True,
        disabled=skip_disabled # Greys out the button
    ):
        with st.spinner("Compiling and verifying your story..."): 
            interview_result = process_interview_step(llm, answers, is_skip=True)
            if interview_result['status'] == 'complete':
                st.session_state.event_text_synthesized = interview_result['final_narrative']
                # On skip, we now trigger an immediate rerun to auto-progress
                st.session_state.page = 'review_narrative' # Auto-route
                st.rerun()
            elif interview_result['status'] == 'error':
                 messages.append(AIMessage(content=interview_result['conversational_response']))
                 st.rerun()
        return

    if user_input := st.chat_input("Your Response:"):
        # Record User's Answer
        messages.append(HumanMessage(content=user_input))
        question_just_answered = st.session_state.next_question
        answers.append({"question": question_just_answered, "answer": user_input})
        
        # Process with LLM for Next Step
        with st.spinner("Processing your response..."): 
            interview_result = process_interview_step(llm, answers)
            st.session_state.next_question = interview_result.get('next_question')
            
            if interview_result['status'] == 'continue':
                messages.append(AIMessage(content=f"{interview_result['conversational_response']} {interview_result['next_question']}"))
            
            elif interview_result['status'] == 'complete':
                st.session_state.event_text_synthesized = interview_result['final_narrative']
                # Auto-progressing even on natural completion
                st.session_state.page = 'review_narrative'
                st.rerun()
            
            elif interview_result['status'] == 'error':
                 messages.append(AIMessage(content=interview_result['conversational_response']))
        
        st.rerun()

def show_narrative_review_page():
    st.title("📝 Review & Confirm Event Description")
    st.markdown("The system has compiled your interview responses into a single, cohesive narrative. Please review and edit the text to ensure it is **accurate and complete**.")
    
    if 'final_event_narrative' not in st.session_state:
        st.session_state.final_event_narrative = st.session_state.event_text_synthesized

    edited_narrative = st.text_area(
        "Your Final, Confirmed Event Narrative:",
        value=st.session_state.final_event_narrative,
        height=300
    )
    
    if st.button("Confirm Narrative and Proceed", type="primary"):
        if len(edited_narrative) < MIN_NARRATIVE_LENGTH:
            st.error(f"Please ensure the narrative is substantial (at least {MIN_NARRATIVE_LENGTH} characters).")
            return
            
        st.session_state.final_event_narrative = edited_narrative
        st.session_state.page = 'situation_rating'
        st.rerun()

def show_situation_rating_page():    
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'situation_motive_scores' not in st.session_state:
        st.session_state.situation_motive_scores = {
            m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL
        }

    st.header("🧐 Situation Appraisal")
    st.info(f"Please rate the importance of the following {len(MOTIVES_FULL)} motives to you based on the event you described **1 = Not Important At All** | **{RATING_SCALE_MAX} = Extremely Important**")

    with st.form("situation_rating_form"):
        motive_scores = st.session_state.situation_motive_scores
        for m in MOTIVES_FULL:
            col1, col2 = st.columns(2) 
            with col1:
                motive_scores[m['motive']]['Promotion'] = st.radio(
                    f"{m['Promotion']}", 
                    options=RADIO_OPTIONS, 
                    index=motive_scores[m['motive']]['Promotion'] - 1, 
                    horizontal=True, 
                    key=f"sit_{m['motive']}_Promotion"
                )
            with col2:
                motive_scores[m['motive']]['Prevention'] = st.radio(
                    f"{m['Prevention']}", 
                    options=RADIO_OPTIONS, 
                    index=motive_scores[m['motive']]['Prevention'] - 1, 
                    horizontal=True, 
                    key=f"sit_{m['motive']}_Prevention"
                )
            st.markdown("<hr style='margin: 0px 0 5px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
            
        if st.form_submit_button("Submit and Finish", type="primary"):
            
            # Build a single trial record and save it.
            trial_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "participant_id": str(uuid.uuid4()),
                "baseline_regulatory_focus": flatten_motive_dict(st.session_state.reg_focus_scores),
                "baseline_motive_profile": flatten_motive_dict(st.session_state.general_motive_scores),
                "interview_history": st.session_state.get('interview_answers', []),
                "confirmed_event_narrative": st.session_state.get('final_event_narrative'),
                "event_situation_rating": flatten_motive_dict(st.session_state.situation_motive_scores),
            }

            # Minimal safety check
            if not trial_data["confirmed_event_narrative"]:
                st.error("Missing event narrative. Please go back and confirm your narrative.")
                return

            with st.spinner("Saving your responses..."):
                if save_data(trial_data):
                    st.session_state.page = 'thank_you'
                    st.rerun()
            
def show_thank_you_page():
    st.title("✅ Trial Complete")
    st.success("Your data has been successfully submitted and saved for the study.")

    if st.button("Start a New Trial", type="primary"):
        for key in list(st.session_state.keys()):
            # Only reset session state keys, not Streamlit secrets/cached resources
            if key not in ['GEMINI_API_KEY', 'gcp_service_account']:
                del st.session_state[key]
        st.session_state.page = 'consent' # Route back to the start
        st.rerun()

# --- 5. MAIN APP EXECUTION ---
if 'page' not in st.session_state:
    st.session_state.page = 'consent' # Start at consent page

# Page routing logic
if st.session_state.page == 'consent':
    show_consent_page()
elif st.session_state.page == 'regulatory':
    show_regulatory_only_page() 
elif st.session_state.page == 'motives':
    show_motives_only_page() 
elif st.session_state.page == 'chat': 
    show_chat_page()
elif st.session_state.page == 'review_narrative':
    show_narrative_review_page()
elif st.session_state.page == 'situation_rating':
    show_situation_rating_page()
elif st.session_state.page == 'thank_you': 
    show_thank_you_page()
