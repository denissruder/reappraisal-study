import streamlit as st
import os
import json
import datetime
import uuid
import time
import random
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

JSON_KEYS_LIST = ", ".join(MOTIVE_SCORE_KEYS)

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

# Indices (0-based) for grouping Regulatory Focus Items (9 Promotion, 9 Prevention)
PROMOTION_ITEMS_0_BASED = [2, 4, 5, 7, 11, 13, 15, 16, 17]
PREVENTION_ITEMS_0_BASED = [0, 1, 3, 6, 8, 9, 10, 12, 14]

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

# RAG Source: Motive Relevance Theory (Minimal f-string use, only for config variables)
MOTIVE_RELEVANCE_RAG = (
f"# Motive Relevance Theory\n"
f"The goal is to predict the **relevance** of a situation to the participant's motive profile. Relevance is rated on a 1 (Not Relevant At All) to {RATING_SCALE_MAX} (Highly Relevant) scale.\n\n"
f"CRITICAL: The output MUST be granular: For each of the {len(MOTIVE_NAMES)} core motives, you MUST predict relevance for two distinct sub-dimensions (Promotion/Prevention). The keys are provided in the input."
)

# FEW-SHOT EXAMPLE for Motive Prediction
APPRAISAL_FEW_SHOT_EXAMPLE = """
# FEW-SHOT EXAMPLE (Consistency Principle)
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
OUTPUT MOTIVE RELEVANCE PREDICTION:
{"motive_relevance_prediction": {"Hedonic_Promotion": 4, "Hedonic_Prevention": 4, "Physical_Promotion": 1, "Physical_Prevention": 1, "Wealth_Promotion": 2, "Wealth_Prevention": 3, "Predictability_Promotion": 6, "Predictability_Prevention": 7, "Competence_Promotion": 9, "Competence_Prevention": 9, "Growth_Promotion": 5, "Growth_Prevention": 4, "Autonomy_Promotion": 3, "Autonomy_Prevention": 5, "Relatedness_Promotion": 4, "Relatedness_Prevention": 6, "Acceptance_Promotion": 7, "Acceptance_Prevention": 8, "Status_Promotion": 7, "Status_Prevention": 8, "Responsibility_Promotion": 7, "Responsibility_Prevention": 9, "Meaning_Promotion": 3, "Meaning_Prevention": 3, "Instrumental_Promotion": 7, "Instrumental_Prevention": 8}}
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

# FIX: Consolidated into one f-string and properly double-escaped all literal braces in the final JSON template to prevent LangChain from reading them as variables.
APPRAISAL_PREDICTION_TEMPLATE = f"""
# PERSONA: APPRAISAL ANALYST (Expert Psychological Assessor)
You are an objective Appraisal Analyst. Your task is to predict the **Motivational Relevance Profile** of the provided situation. This task adheres to **ConVe principles** (Consistency and Verifiability).

{MOTIVE_RELEVANCE_RAG}

{APPRAISAL_FEW_SHOT_EXAMPLE}

--- TASK INSTRUCTIONS ---
1. **Analyze:** Carefully review the SITUATION DESCRIPTION below.
2. **Chain-of-Thought (CoT):** You MUST provide your step-by-step reasoning within a <REASONING> block.
3. **Rating Scale:** Use the 1 (Not Relevant At All) to {RATING_SCALE_MAX} (Highly Relevant) scale.
4. **Output Format:** Output MUST be a valid JSON object. The JSON MUST contain a single top-level key, "motive_relevance_prediction", whose value is a dictionary containing **all 26 keys** listed below, with a score from 1 to {RATING_SCALE_MAX}.

REQUIRED JSON KEYS (26 total): {{required_keys}}

--- TASK INPUT ---
SITUATION DESCRIPTION: {{event_text}}

--- YOUR PREDICTION ---
Begin the analysis below.

<REASONING>
1. [Analyze Event Impact on Motive 1: Promotion, Prevention]
2. [Analyze Event Impact on Motive 2: Promotion, Prevention]
3. ...
</REASONING>

OUTPUT MOTIVE RELEVANCE PREDICTION:
{{"motive_relevance_prediction": {{"Hedonic_Promotion": 5, "Hedonic_Prevention": 5, "Physical_Promotion": 5, "Physical_Prevention": 5, "Wealth_Promotion": 5, "Wealth_Prevention": 5, "Predictability_Promotion": 5, "Predictability_Prevention": 5, "Competence_Promotion": 5, "Competence_Prevention": 5, "Growth_Promotion": 5, "Growth_Prevention": 5, "Autonomy_Promotion": 5, "Autonomy_Prevention": 5, "Relatedness_Promotion": 5, "Relatedness_Prevention": 5, "Acceptance_Promotion": 5, "Acceptance_Prevention": 5, "Status_Promotion": 5, "Status_Prevention": 5, "Responsibility_Promotion": 5, "Responsibility_Prevention": 5, "Meaning_Promotion": 5, "Meaning_Prevention": 5, "Instrumental_Promotion": 5, "Instrumental_Prevention": 5}}}}
"""

def parse_llm_json(response_content, attempt_number=0):
    """Safely extracts and parses the JSON block from the LLM's response."""
    
    json_string = response_content.strip()
    
    # --- NEW: Improved Heuristics for JSON Extraction ---
    
    # 1. Prioritize stripping markdown code blocks (```json ... ```)
    if json_string.startswith("```"):
        # Find the first and last triple-backtick markers
        start_index = json_string.find("```")
        end_index = json_string.rfind("```")
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_string = json_string[start_index + 3:end_index].strip()
            # Remove optional language specifier (e.g., 'json')
            if json_string.lower().startswith('json'):
                 json_string = json_string[4:].strip()

    # 2. Fallback: Find the last opening brace '{' and assume JSON starts there
    # This is necessary for responses that omit the markdown block wrappers
    if not (json_string.startswith("{") and json_string.endswith("}")):
         last_open_brace = json_string.rfind("{")
         if last_open_brace != -1:
             json_string = json_string[last_open_brace:].strip()
             # Attempt to clean up trailing text if the JSON object ends early
             last_close_brace = json_string.rfind("}")
             if last_close_brace != -1 and last_close_brace < len(json_string) - 1:
                  json_string = json_string[:last_close_brace + 1]

    try:
        # Check if the remaining string is parseable
        analysis_data = json.loads(json_string, strict=True)
        
        # ... (Keep the validation checks for 'motive_relevance_prediction' key count, and score range) ...
                
        # Return the actual scores dictionary for aggregation
        return {k: int(round(float(v))) for k, v in prediction_scores.items()}
            
    except json.JSONDecodeError as e:
        # --- DEBUG STEP 3: Display raw content on JSON Decode failure ---
        st.error(f"Parse Error (Attempt {attempt_number}): JSON decoding failed. Error: {e}")
        with st.expander(f"❌ DEBUG: View Raw Content that Failed to Parse (Attempt {attempt_number})"):
             st.code(response_content, language="text") # Show the full, uncleaned LLM output
        # --- END DEBUG STEP 3 ---
        return None
        
    except Exception as e:
        st.error(f"Parse Error (Attempt {attempt_number}): General exception during parsing: {e}")
        return None
            
    except json.JSONDecodeError as e:
        st.error(f"Parse Error: JSON decoding failed. Response content starts with: {json_string[:100]}... Error: {e}")
        return None
    except Exception as e:
        st.error(f"Parse Error: General exception during parsing: {e}")
        return None
        
@st.cache_data(show_spinner=False)
def run_self_consistent_appraisal_prediction(llm_instance, event_text):
    """
    Executes the LLM N_COTS times to generate a self-consistent prediction.
    Returns the aggregated prediction or (None, last_failed_response_content) on failure.
    """
    
    prompt = PromptTemplate(
        # CRITICAL FIX: Include the new input variable "required_keys"
        input_variables=["event_text", "required_keys"], 
        template=APPRAISAL_PREDICTION_TEMPLATE
    )
    chain = prompt | llm_instance
    
    # --- DEBUG STEP 1: Capture and Display the FULL Prompt ---
    try:
        # Format the prompt with both variables for debugging visibility
        final_prompt_content = prompt.format(event_text=event_text, required_keys=JSON_KEYS_LIST)
        with st.expander("🔍 DEBUG: View Final Prompt Sent to LLM"):
            st.code(final_prompt_content, language="markdown")
    except Exception as e:
        st.error(f"DEBUG ERROR: Failed to format final prompt: {e}")
    # --- END DEBUG STEP 1 ---
    
    valid_predictions = []
    last_failed_response = "" 

    # Run the model multiple times (Self-Consistency)
    for i in range(N_COTS):
        try:
            st.info(f"Generating prediction attempt {i+1}/{N_COTS}...")
            
            # CRITICAL FIX: Invoke chain with both variables
            response = chain.invoke({
                "event_text": event_text, 
                "required_keys": JSON_KEYS_LIST
            })
            
            response_content = response.content
            
            if not response_content:
                st.warning(f"DEBUG: Attempt {i+1} received an empty response.")
                continue

            # Pass the raw response to the parser
            parsed_scores = parse_llm_json(response_content, attempt_number=i+1)

            if parsed_scores:
                valid_predictions.append(parsed_scores)
                st.success(f"Prediction {i+1} successful and valid.")
            else:
                st.warning(f"Prediction {i+1} failed validation (parsing/structure error). Skipping.")
                last_failed_response = response_content # Update on failure
                
        except Exception as e:
            st.error(f"Error during LLM Appraisal Prediction run {i+1} (Invocation failed): {e}")
        
        time.sleep(0.5) 

    if not valid_predictions:
        st.error(f"❌ Final Failure: All {N_COTS} LLM attempts failed to produce a valid prediction.")
        return (None, last_failed_response) 

    # Aggregation step (Self-Consistency)
    final_prediction = {}
    for key in MOTIVE_SCORE_KEYS:
        scores = [p[key] for p in valid_predictions]
        final_prediction[key] = int(round(sum(scores) / len(scores)))

    return ({"motive_relevance_prediction": final_prediction, "n_cots_used": len(valid_predictions)}, "")
    
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
        
        # --- PREVENTION FOCUS ITEMS (Avoiding negative outcomes) ---
        st.markdown("#### 1. Prevention Focus (Focus on obligations, safety, and avoiding negative outcomes)")
        for i in PREVENTION_ITEMS_0_BASED:
            item = REG_FOCUS_ITEMS[i]
            st.session_state.reg_focus_scores[item] = st.radio(
                f"**{i+1}.** {item}", 
                options=RADIO_OPTIONS, 
                index=reg_focus_scores[item] - 1, 
                horizontal=True, 
                key=f"reg_focus_{i}"
            )
            
        # --- PROMOTION FOCUS ITEMS (Achieving positive outcomes) ---
        st.markdown("#### 2. Promotion Focus (Focus on hopes, achievements, and positive outcomes)")
        for i in PROMOTION_ITEMS_0_BASED:
            item = REG_FOCUS_ITEMS[i]
            st.session_state.reg_focus_scores[item] = st.radio(
                f"**{i+1}.** {item}", 
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

def show_cross_rating_page():
    """Renders the cross-participant rating task (Step 4) and triggers the Self-Consistent LLM prediction (Step 5)."""
    st.title("👥 Cross-Participant Appraisal (26 Scores)")
    st.markdown("Finally, please read the situation described by **another participant** and complete the same relevance questionnaire from what you believe was **their perspective**.")

    # --- Fetch a random story from the DB ---
    random_situation = get_random_story_from_db()
    
    st.subheader("Situation from Another Participant:")
    with st.container(border=True):
        st.info(random_situation)

    # Define the 1-9 radio options
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 
    
    if 'cross_motive_scores' not in st.session_state:
        st.session_state.cross_motive_scores = {
            m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL
        }

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
            
            # --- Run LLM Prediction Silently and Early ---
            llm_prediction_result = None
            last_failed_response = ""
            with st.spinner("Finalizing study submission (Running system checks in background)..."):
                 # The function now returns a tuple: (result, last_failed_response)
                 llm_prediction_result, last_failed_response = run_self_consistent_appraisal_prediction(llm, st.session_state.final_event_narrative)
            
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
                 st.error("Submission failed. The system was unable to generate a valid prediction after multiple attempts. This usually means the LLM output was malformed (e.g., non-JSON, missing keys, or scores out of range).")
                 
                 if last_failed_response:
                     st.subheader("Last Failed LLM Response (for debugging):")
                     # Display the raw response that failed the parsing
                     st.code(last_failed_response, language="json")
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
