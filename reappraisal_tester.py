import streamlit as st
import os
import json
import datetime
import uuid
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 0. Streamlit UI Setup ---

# Set wide layout and title for better form visibility
st.set_page_config(layout="wide", page_title="Version A: Appraisal Prediction Study (V4 - Self-Consistency)")

# Inject minimal CSS for a cleaner look
st.markdown("""
<style>
/* Adjust container width for forms */
.stForm {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}
/* Style for motive headers in forms */
.stForm h4 {
    margin-top: 20px;
    padding-bottom: 5px;
    border-bottom: 1px solid #ddd;
}
/* Reduce space between radio options for horizontal layout */
div[data-testid="stForm"] label {
    margin-right: 15px;
}
</style>
""", unsafe_allow_html=True)


# --- 1. CONFIGURATION & SETUP ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.8 # Increased temperature for more diverse CoTs
RATING_SCALE_MAX = 9 
MIN_NARRATIVE_LENGTH = 100
N_COTS = 5 # *** NEW: Number of Chain-of-Thought runs for Self-Consistency ***

# Comprehensive list of Motives and their Promotion/Prevention framings
MOTIVES_FULL = [
    {"motive": "Hedonic", "promotion": "To feel good", "prevention": "Not to feel bad"},
    {"motive": "Physical", "promotion": "To be in good health", "prevention": "To stay safe"},
    {"motive": "Wealth", "promotion": "To have money", "prevention": "To avoid poverty"},
    {"motive": "Predictability", "promotion": "To understand", "prevention": "To avoid confusion"},
    {"motive": "Competence", "promotion": "To succeed", "prevention": "To avoid failure"},
    {"motive": "Growth", "promotion": "To learn and grow", "prevention": "To avoid monotony or decline"},
    {"motive": "Autonomy", "promotion": "To be free to decide", "prevention": "Not to be told what to do"},
    {"motive": "Relatedness", "promotion": "To feel connected", "prevention": "To avoid loneliness"},
    {"motive": "Acceptance", "promotion": "To be liked", "prevention": "To avoid disapproval"},
    {"motive": "Status", "promotion": "To stand out", "prevention": "To avoid being ignored"},
    {"motive": "Responsibility", "promotion": "To live up to expectations", "prevention": "Not to let others down"},
    {"motive": "Meaning", "promotion": "To make a difference", "prevention": "Not to waste my life"},
    {"motive": "Instrumental", "promotion": "To gain something", "prevention": "To avoid something"},
]
MOTIVE_NAMES = [m["motive"] for m in MOTIVES_FULL] # List of 13 motives

# Keys for the 26 final scores
MOTIVE_SCORE_KEYS = [
    f"{m['motive']}_{dim}"
    for m in MOTIVES_FULL for dim in ['Promotion', 'Prevention']
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

# --- RAG Contexts and Few-Shot Examples ---

# RAG Source: Motive Relevance Theory
MOTIVE_RELEVANCE_RAG = f"""
# Motive Relevance Theory and Granular Assessment
The goal is to predict the **relevance** of a situation to the participant's motive profile. Relevance is rated on a 1 (Not Relevant At All) to {RATING_SCALE_MAX} (Highly Relevant) scale.

CRITICAL: For each of the {len(MOTIVE_NAMES)} core motives, you MUST predict relevance for two distinct sub-dimensions:
1. **Promotion:** Relevance to the **growth/gain** component (e.g., To succeed/To feel good).
2. **Prevention:** Relevance to the **safety/loss** component (e.g., To avoid failure/Not to feel bad).

Your final JSON output must contain **26** key-value pairs (13 Motives * 2 Dimensions).
"""

# FEW-SHOT EXAMPLE for Motive Prediction
APPRAISAL_FEW_SHOT_EXAMPLE = f"""
# FEW-SHOT EXAMPLE: High Competence and Status Relevance
INPUT SITUATION: "I was publicly criticized by my boss for a mistake I made on a major project. I feel intense shame and worry about my job."
---
<REASONING>
1. **Competence Analysis:** Public failure is a direct threat.
   - Promotion (To succeed): High relevance because the failure means a loss of success (9).
   - Prevention (To avoid failure): Extremely high relevance because the fear of failure was realized (9).
2. **Status Analysis:** Public criticism directly impacts social standing.
   - Promotion (To stand out): Hindered, so highly relevant (7).
   - Prevention (To avoid being ignored): Hindered, fear of negative attention was realized (8).
3. **Physical Analysis:** No direct physical impact. All scores low (1).
</REASONING>
OUTPUT MOTIVE RELEVANCE PREDICTION (JSON block follows immediately after reasoning):
{{"motive_relevance_prediction": {{
"Hedonic_Promotion": 4, "Hedonic_Prevention": 4,
"Physical_Promotion": 1, "Physical_Prevention": 1,
"Wealth_Promotion": 2, "Wealth_Prevention": 3,
"Predictability_Promotion": 6, "Predictability_Prevention": 7,
"Competence_Promotion": 9, "Competence_Prevention": 9,
"Growth_Promotion": 5, "Growth_Prevention": 4,
"Autonomy_Promotion": 3, "Autonomy_Prevention": 5,
"Relatedness_Promotion": 4, "Relatedness_Prevention": 6,
"Acceptance_Promotion": 7, "Acceptance_Prevention": 8,
"Status_Promotion": 7, "Status_Prevention": 8,
"Responsibility_Promotion": 7, "Responsibility_Prevention": 9,
"Meaning_Promotion": 3, "Meaning_Prevention": 3,
"Instrumental_Promotion": 7, "Instrumental_Prevention": 8
}}}}
"""

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
COLLECTION_NAME = "version_a_appraisal_trials_v4"

# --- NEW: Function to fetch a random story from the DB ---
@st.cache_data(ttl=600) # Cache for 10 minutes to avoid hitting DB too often
def get_random_story_from_db():
    """Fetches a random 'confirmed_event_narrative' from the Firestore collection."""
    try:
        # Fetch up to 100 documents (adjust limit if needed for a very large collection)
        # We only need the 'confirmed_event_narrative' field.
        docs = db.collection(COLLECTION_NAME).select(['confirmed_event_narrative']).limit(100).get()
        
        stories = [
            doc.get('confirmed_event_narrative')
            for doc in docs
            if doc.get('confirmed_event_narrative') and len(doc.get('confirmed_event_narrative')) >= MIN_NARRATIVE_LENGTH
        ]
        
        if stories:
            return random.choice(stories)
        else:
            return "No complete event narratives found in the database. Using a fallback placeholder."
            
    except Exception as e:
        st.warning(f"Failed to fetch random story from database: {e}. Using a fallback placeholder.")
        return "Database fetch failed. Placeholder: I was supposed to give a major presentation to my client, and just minutes before, my laptop crashed, losing several hours of preparation. I had to improvise everything on a backup system. I felt incompetent, and worried I would lose the client's business, which would reflect poorly on my whole team."

# --- 2. LLM LOGIC FUNCTIONS (Self-Consistency/Multiple CoTs) ---

# --- LLM APPRAISAL PREDICTION TEMPLATE ---
APPRAISAL_PREDICTION_TEMPLATE = f"""
# ROLE: APPRAISAL ANALYST (Expert Psychological Assessor)
You are an objective Appraisal Analyst. Your task is to predict the **Motivational Relevance Profile** of the provided situation. This prediction must be highly granular, adhering to the 13 Motives and their 2 sub-dimensions (Promotion/Prevention).

CRITICAL INSTRUCTIONS:
1. **Use RAG Context:** Adhere to the {RATING_SCALE_MAX}-point scale and the granular definition of Relevance provided.
2. **Chain-of-Thought (CoT):** You MUST provide your step-by-step reasoning within a <REASONING> block before providing the final JSON output.
3. **Output Format:** The final JSON MUST contain **26** motive relevance scores, using the exact key format: `[MotiveName]_Promotion` and `[MotiveName]_Prevention`.

# RAG CONTEXT: MOTIVE RELEVANCE THEORY
{MOTIVE_RELEVANCE_RAG}

# FEW-SHOT EXAMPLE
{APPRAISAL_FEW_SHOT_EXAMPLE}

--- TASK INPUT ---
SITUATION DESCRIPTION: {{event_text}}

--- YOUR PREDICTION ---
Begin the analysis below.

<REASONING>
1. [Analyze Event Impact on Motive 1: Promotion, Prevention]
2. [Analyze Event Impact on Motive 2: Promotion, Prevention]
3. ...
</REASONING>

Provide the JSON output (Only the JSON block should follow the </REASONING> tag):
"""

def parse_llm_json(response_content):
    """Safely extracts and parses the JSON block from the LLM's response."""
    try:
        json_string = response_content.strip()
        
        # Heuristics to find the JSON start and end
        if json_string.rfind("{") != -1:
            json_string = json_string[json_string.rfind("{"):].strip()
        
        if json_string.startswith("```json"):
            json_string = json_string.lstrip("```json").rstrip("```")
        elif json_string.startswith("```"):
            json_string = json_string.lstrip("```").rstrip("```")

        analysis_data = json.loads(json_string, strict=False)
        
        if 'motive_relevance_prediction' not in analysis_data:
             # --- LOGGING FOR DEBUG ---
             print(f"Parse Error: Missing 'motive_relevance_prediction' key in top level JSON.")
             return None # Missing required top-level key
             
        prediction_scores = analysis_data['motive_relevance_prediction']
        
        # Validate that we have the correct number of keys (26)
        if len(prediction_scores) != len(MOTIVE_SCORE_KEYS):
            # --- LOGGING FOR DEBUG ---
            print(f"Parse Error: Incorrect number of keys. Expected 26, got {len(prediction_scores)}.")
            return None
            
        # Ensure all scores are integers/floats and within the 1-9 range
        for key in MOTIVE_SCORE_KEYS:
            # Safely check key existence and range constraint
            if key not in prediction_scores:
                # --- LOGGING FOR DEBUG ---
                print(f"Parse Error: Missing required key '{key}'.")
                return None
            
            try:
                score = float(prediction_scores[key])
                if not (1 <= score <= RATING_SCALE_MAX):
                    # --- LOGGING FOR DEBUG ---
                    print(f"Parse Error: Score for '{key}' is {score}, which is outside the range 1-{RATING_SCALE_MAX}.")
                    return None
            except ValueError:
                # --- LOGGING FOR DEBUG ---
                print(f"Parse Error: Score for '{key}' is not a valid number.")
                return None
                
        # Return the actual scores dictionary for aggregation
        return {k: int(round(float(v))) for k, v in prediction_scores.items()}
        
    except json.JSONDecodeError as e:
        # --- LOGGING FOR DEBUG ---
        print(f"Parse Error: JSON decoding failed. Response content starts with: {response_content[:100]}... Error: {e}")
        return None
    except Exception as e:
        # --- LOGGING FOR DEBUG ---
        print(f"Parse Error: General exception during parsing: {e}")
        return None


@st.cache_data(show_spinner=False)
def run_self_consistent_appraisal_prediction(llm_instance, event_text):
    """
    Executes the LLM N_COTS times to generate a self-consistent prediction.
    *** MODIFIED: Removed all st.info/st.success/st.error calls to hide progress. ***
    """
    
    prompt = PromptTemplate(
        input_variables=["event_text"], 
        template=APPRAISAL_PREDICTION_TEMPLATE
    )
    chain = prompt | llm_instance
    
    valid_predictions = []
    
    # Run the model multiple times (Self-Consistency)
    for i in range(N_COTS):
        # st.info(f"Generating prediction attempt {i+1}/{N_COTS}...") # REMOVED
        try:
            # Using st.cache_data means this function must be deterministic, 
            # so we use LLM's built-in temperature/stochasticity and a simple loop.
            response = chain.invoke({"event_text": event_text})
            parsed_scores = parse_llm_json(response.content)
            
            if parsed_scores:
                valid_predictions.append(parsed_scores)
                # st.success(f"Prediction {i+1} successful and valid.") # REMOVED
            else:
                # st.warning(f"Prediction {i+1} failed validation (parsing/structure error). Skipping.") # REMOVED
                pass
                
        except Exception:
            # st.error(f"Error during LLM Appraisal Prediction run {i+1}: {e}") # REMOVED
            pass
        
        # Small delay to potentially aid in CoT diversity
        time.sleep(0.5) 

    if not valid_predictions:
        # st.error(f"All {N_COTS} LLM attempts failed to produce a valid prediction. Cannot proceed.") # REMOVED
        return None

    # Aggregation step (Self-Consistency)
    final_prediction = {}
    for key in MOTIVE_SCORE_KEYS:
        # Collect all scores for the current key from valid runs
        scores = [p[key] for p in valid_predictions]
        # Calculate the mean (rounding to the nearest integer as the original scale is 1-9)
        final_prediction[key] = int(round(sum(scores) / len(scores)))

    # st.success(f"✅ Self-Consistency complete. Aggregated result from {len(valid_predictions)} valid CoT runs.") # REMOVED
    return {"motive_relevance_prediction": final_prediction, "n_cots_used": len(valid_predictions)}

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


# --- 3. DATA SAVING LOGIC (No change) ---

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
    st.title("🎯 Initial Assessment: Regulatory Focus")
    
    # Define the 1-9 radio options
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'reg_focus_scores' not in st.session_state:
        st.session_state.reg_focus_scores = {item: 5 for item in REG_FOCUS_ITEMS}

    with st.form("regulatory_assessment_form"):
        st.subheader("Regulatory Focus (General Tendency)")
        st.markdown(f"Please indicate how true the following {len(REG_FOCUS_ITEMS)} statements are of you **in general** on a scale of 1 to {RATING_SCALE_MAX}.")
        st.markdown(f"**1 = Not At All True of Me** | **{RATING_SCALE_MAX} = Very True of Me**")
        
        reg_focus_scores = st.session_state.reg_focus_scores
        for i, item in enumerate(REG_FOCUS_ITEMS):
            # Item prefix removed, using the statement directly
            st.session_state.reg_focus_scores[item] = st.radio(
                f"**{item}**", 
                options=RADIO_OPTIONS, 
                index=reg_focus_scores[item] - 1, 
                horizontal=True, 
                key=f"reg_focus_{i}"
            )

        if st.form_submit_button("Next: General Motive Profile", type="primary"):
            st.session_state.page = 'motives' # Route to motives next
            st.rerun()

def show_motives_only_page():
    st.title("🎯 Initial Assessment: General Motive Profile")
    
    # Define the 1-9 radio options
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'general_motive_scores' not in st.session_state:
        st.session_state.general_motive_scores = {
            m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL
        }

    with st.form("initial_assessment_form"):
        st.subheader("General Motive Importance & Focuses")
        st.markdown(f"""
        Please rate the importance of the following {len(MOTIVES_FULL)} motives to you **in general** on a scale of 1 to {RATING_SCALE_MAX}. 
        You must provide **two scores** for each motive: Promotion Focus and Prevention Focus.
        """)
        st.markdown(f"**1 = Not Important At All** | **{RATING_SCALE_MAX} = Extremely Important**")

        motive_scores = st.session_state.general_motive_scores
        for m in MOTIVES_FULL:
            st.markdown(f"#### Motive: {m['motive']}")
            
            # Promotion Focus (NOW RADIO BUTTONS)
            motive_scores[m['motive']]['Promotion'] = st.radio(
                f"Promotion Focus: *{m['promotion']}*",
                options=RADIO_OPTIONS, 
                index=motive_scores[m['motive']]['Promotion'] - 1, 
                horizontal=True, 
                key=f"gen_{m['motive']}_Promotion"
            )
            # Prevention Focus (NOW RADIO BUTTONS)
            motive_scores[m['motive']]['Prevention'] = st.radio(
                f"Prevention Focus: *{m['prevention']}*",
                options=RADIO_OPTIONS, 
                index=motive_scores[m['motive']]['Prevention'] - 1, 
                horizontal=True, 
                key=f"gen_{m['motive']}_Prevention"
            )

        if st.form_submit_button("Next: Start Interview", type="primary"):
            st.session_state.page = 'chat' # Route to chat next
            st.rerun()


def show_chat_page():
    st.header("🗣️ Event Interview")
    st.markdown("Please describe a recent emotionally unpleasant event. The chatbot will ask follow-up questions to gather necessary context. **You can stop the interview at any time by clicking the button below.**")

    if 'interview_messages' not in st.session_state:
        # Initialize with the first question directly
        initial_question = INTERVIEW_QUESTIONS[0]
        st.session_state.interview_messages = [AIMessage(content=initial_question)]
        st.session_state.interview_answers = []
        st.session_state.next_question = initial_question # Use this to track the question just asked
        st.session_state.event_text_synthesized = None

    messages = st.session_state.interview_messages
    answers = st.session_state.interview_answers
    
    chat_container = st.container(height=450, border=True)

    with chat_container:
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    if st.session_state.event_text_synthesized:
        st.success("✅ Interview complete. Proceed to the next step.")
        if st.button("Next: Review and Confirm Narrative", type="primary", use_container_width=True):
            st.session_state.page = 'review_narrative'
            st.rerun()
        return
        
    # Manual skip button
    if st.button("Skip to Narrative Synthesis", type="secondary", use_container_width=True):
        if not answers:
             st.error("Please provide at least one response before synthesizing the narrative.")
             return
             
        # Force synthesis
        with st.spinner("Compiling and verifying your story..."): 
            interview_result = process_interview_step(llm, answers, is_skip=True)
            if interview_result['status'] == 'complete':
                st.session_state.event_text_synthesized = interview_result['final_narrative']
                messages.append(AIMessage(content=interview_result['conversational_response']))
            elif interview_result['status'] == 'error':
                 # Error message is already displayed in the function
                 messages.append(AIMessage(content=interview_result['conversational_response']))
            st.rerun()
        return

    if user_input := st.chat_input("Your Response:"):
        
        # 1. Record User's Answer
        messages.append(HumanMessage(content=user_input))
        
        # The question just answered is the one tracked by st.session_state.next_question
        question_just_answered = st.session_state.next_question
        answers.append({"question": question_just_answered, "answer": user_input})
        
        # 2. Process with LLM for Next Step
        with st.spinner("Processing your response..."): 
            interview_result = process_interview_step(llm, answers)
            
            st.session_state.next_question = interview_result.get('next_question')
            
            if interview_result['status'] == 'continue':
                # Continue the interview
                messages.append(AIMessage(content=f"{interview_result['conversational_response']} {interview_result['next_question']}"))
            
            elif interview_result['status'] == 'complete':
                # Interview is complete, save the narrative
                st.session_state.event_text_synthesized = interview_result['final_narrative']
                messages.append(AIMessage(content=interview_result['conversational_response']))
            
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
    st.title("📊 Situation Appraisal: Your Perspective (26 Scores)")
    st.markdown(f"""
    Please rate how **relevent** each of the following motives and their focuses was to the **event you just described** on a scale of 1 to {RATING_SCALE_MAX}.
    
    **1 = Not Relevant At All** | **{RATING_SCALE_MAX} = Extremely Relevant** (The situation strongly helped OR hindered this motive/focus.)
    """)
    
    # Define the 1-9 radio options
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'situation_motive_scores' not in st.session_state:
        st.session_state.situation_motive_scores = {
            m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL
        }

    with st.form("situation_rating_form"):
        situation_scores = st.session_state.situation_motive_scores
        
        for m in MOTIVES_FULL:
            st.markdown(f"#### Motive: {m['motive']}")
            
            # Promotion Focus Relevance (NOW RADIO BUTTONS)
            situation_scores[m['motive']]['Promotion'] = st.radio(
                f"Relevance to Promotion Focus: *{m['promotion']}*",
                options=RADIO_OPTIONS, 
                index=situation_scores[m['motive']]['Promotion'] - 1, 
                horizontal=True, 
                key=f"sit_{m['motive']}_Promotion"
            )
            # Prevention Focus Relevance (NOW RADIO BUTTONS)
            situation_scores[m['motive']]['Prevention'] = st.radio(
                f"Relevance to Prevention Focus: *{m['prevention']}*",
                options=RADIO_OPTIONS, 
                index=situation_scores[m['motive']]['Prevention'] - 1, 
                horizontal=True, 
                key=f"sit_{m['motive']}_Prevention"
            )

        if st.form_submit_button("Next: Cross-Participant Rating", type="primary"):
            st.session_state.page = 'cross_rating'
            st.rerun()

  with st.form("cross_rating_form"):
        st.markdown(f"""
        Please rate the relevance of the situation above on a scale of 1 to {RATING_SCALE_MAX}, 
        based on what you think the **original author felt** (their perspective). You must provide **two scores** for each motive.
        """)

        cross_scores = st.session_state.cross_motive_scores
        for m in MOTIVES_FULL:
            st.markdown(f"#### Motive: {m['motive']}")
            
            # Promotion Focus Relevance (NOW RADIO BUTTONS)
            cross_scores[m['motive']]['Promotion'] = st.radio(
                f"Relevance to Promotion Focus: *{m['promotion']}* (The author's perspective)",
                options=RADIO_OPTIONS, 
                index=cross_scores[m['motive']]['Promotion'] - 1, 
                horizontal=True, 
                key=f"cross_{m['motive']}_Promotion"
            )
            # Prevention Focus Relevance (NOW RADIO BUTTONS)
            cross_scores[m['motive']]['Prevention'] = st.radio(
                f"Relevance to Prevention Focus: *{m['prevention']}* (The author's perspective)",
                options=RADIO_OPTIONS, 
                index=cross_scores[m['motive']]['Prevention'] - 1, 
                horizontal=True, 
                key=f"cross_{m['motive']}_Prevention"
            )
        
        if st.form_submit_button("Submit All Data and Finish Trial", type="primary"):
            st.session_state.cross_participant_situation = random_situation
            
            # --- MODIFIED: Run LLM Prediction Silently and Early ---
            # We use a silent spinner to hide the "Self-Consistency in Progress" phase
            # while the actual prediction runs in the background.
            llm_prediction_result = None
            with st.spinner("Finalizing study submission..."):
                 # The function itself has been modified to remove internal st.info/st.success calls.
                 llm_prediction_result = run_self_consistent_appraisal_prediction(llm, st.session_state.final_event_narrative)
            
            if llm_prediction_result:
                trial_data = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "participant_id": str(uuid.uuid4()), 
                    
                    # Step 1 Data (26 baseline scores)
                    "baseline_motive_profile": st.session_state.general_motive_scores,
                    "baseline_regulatory_focus": st.session_state.reg_focus_scores,
                    
                    # Step 2 Data
                    "interview_qa_history": st.session_state.interview_answers, 
                    "confirmed_event_narrative": st.session_state.final_event_narrative,
                    
                    # LLM Prediction Data (The LLM's Hypothesis - Self-Consistent 26 scores)
                    "llm_motive_relevance_prediction": llm_prediction_result.get('motive_relevance_prediction'),
                    "llm_n_cots_used": llm_prediction_result.get('n_cots_used'),
                    
                    # Step 3 Data (Original Author's Target - 26 scores)
                    "situation_motive_relevance_rating": st.session_state.situation_motive_scores,
                    
                    # Step 4 Data (Comparison Target - 26 scores)
                    "cross_participant_situation": st.session_state.cross_participant_situation,
                    "cross_participant_motive_rating": st.session_state.cross_motive_scores,
                }
                
                if save_data(trial_data):
                    st.session_state.page = 'thank_you'
                else:
                    return # Keep user on the page if save failed
            else:
                 st.error("Submission failed. The system was unable to generate a valid prediction.")
                 return
            
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
    show_regulatory_only_page() # New first assessment page
elif st.session_state.page == 'motives':
    show_motives_only_page() # New second assessment page
elif st.session_state.page == 'chat': 
    show_chat_page()
elif st.session_state.page == 'review_narrative':
    show_narrative_review_page()
elif st.session_state.page == 'situation_rating':
    show_situation_rating_page()
elif st.session_state.page == 'cross_rating':
    show_cross_rating_page()
elif st.session_state.page == 'thank_you': 
    show_thank_you_page()

# This ensures the user is always at the top of the new page, which is helpful
# when navigating from a long page (like a rating form) to a new one.
st.markdown(
    """
    <script>
        window.scrollTo(0, 0);
    </script>
    """,
    unsafe_allow_html=True
)
