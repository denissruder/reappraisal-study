import streamlit as st
import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- Inject Custom CSS to HIDE the Streamlit Sidebar Elements ---
# This CSS targets the navigation (the hamburger menu) and the sidebar wrapper,
# ensuring a clean, fullscreen view for the participant by removing the sidebar controls.
st.markdown(
    """
    <style>
    /* Hide the sidebar button (hamburger menu) */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide the top-right button to re-open the sidebar */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Ensure the main content uses the available width for the centered layout */
    .main {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Set Streamlit page configuration ---
# Removing 'layout="wide"' to revert to the default, narrower, centered page layout.
st.set_page_config()

# --- 1. CONFIGURATION & SETUP ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.5 
MOTIVES = ["Achievement & Success", "Security & Stability", "Affiliation & Belonging", "Stimulation & Excitement", "Self-Direction & Autonomy"]
RATING_DIMENSIONS = ["Believability", "Appropriateness", "Emotional Valence"]
MOTIVE_SCALE_MAX = 7 # All ratings will be on a 1-7 scale
MIN_EVENT_LENGTH = 200 # Minimum number of characters required for the event description
CONDITION_OPTIONS = ["1. Neutral", "2. Appraisal-Assessed", "3. Appraisal-Aware"] # All conditions to be run
GUIDANCE_LABELS = ["Guidance 1", "Guidance 2", "Guidance 3"] # User-facing labels

# --- Interview Questions (The mandatory content points) ---
INTERVIEW_QUESTIONS = [
    "What happened? Describe this as if you were telling the story to a friend.", # Initial prompt / narrative
    "Who else was around or involved?", # Other people
    "What else should we know about the situation? What was the context?", # Context/Background
    "What was the outcome?", # Outcome
    "Why did this situation matter to you? What made it important?", # Importance/Relevance
    "Some situations are predictable and clear, others less so. How about this one, was it predictable in any way? Why?", # Predictability
    "Who could be considered responsible for this situation happening?", # Responsibility
    "Would you say you were in control of this situation? Why?", # Control
    "What would you have wanted to happen in this situation, if you could change anything?", # Desired outcome/Discrepancy 1
    "How big of a deal this discrepancy is for you?", # Discrepancy 2
]

try:
    from google.cloud import firestore
except ImportError:
    st.error("The 'google-cloud-firestore' package is not installed. Please check requirements.txt.")
    st.stop()

# 1.1 Securely load Firebase credentials and initialize Firestore client
@st.cache_resource
def get_firestore_client():
    if "gcp_service_account" not in st.secrets:
        st.error("Database Error: 'gcp_service_account' secret not found.")
        st.stop()
        
    try:
        key_dict = json.loads(st.secrets["gcp_service_account"], strict=False)
        db = firestore.Client.from_service_account_info(key_dict)
        # Status message kept for execution tracking, although hidden by CSS
        st.sidebar.success("✅ Database Connected") 
        return db
    except Exception as e:
        st.sidebar.error(f"❌ Database Connection Failed: {e}")
        st.stop()
        
db = get_firestore_client()
COLLECTION_NAME = "reappraisal_study_trials"


# 1.2 Securely load Gemini API key and initialize LLM
@st.cache_resource
def get_llm():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("LLM Error: 'GEMINI_API_KEY' secret not found.")
        st.stop()

    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)
        # Status message kept for execution tracking, although hidden by CSS
        st.sidebar.success("✅ LLM Initialized") 
        return llm
    except Exception as e:
        st.sidebar.error(f"❌ LLM Initialization Failed: {e}")
        st.stop()

llm = get_llm()


# --- 2. LLM LOGIC FUNCTIONS ---

# --- LLM APPRAISAL ANALYSIS (No change) ---
APPRAISAL_ANALYSIS_TEMPLATE = f"""
You are an Appraisal Analyst. Your task is to analyze the user's event description in the context of their core motives.
Your output MUST be a valid JSON object. Do not include any text, headers, or markdown formatting outside of the JSON block.

The JSON object MUST contain a single field:
1. "congruence_ratings": A dictionary where keys are the motive names from the MOTIVE LIST and values are a score from 1 (Low Congruence) to {MOTIVE_SCALE_MAX} (High Congruence). This score represents how congruent the event is with each specific motive.

MOTIVE LIST: {{motive_list}}
User's Event Description: {{event_text}}
User's Motive Importance Scores: {{scores_list_formatted}}

Provide the JSON output:
"""

@st.cache_data(show_spinner=False)
def run_appraisal_analysis(llm_instance, motive_scores, event_text):
    """Executes the LLM to perform Appraisal Analysis."""
    
    scores_list_formatted = "\n".join([f"- {motive}: {score}/{MOTIVE_SCALE_MAX}" for motive, score in motive_scores.items()])
    motive_list = ", ".join(MOTIVES)

    prompt = PromptTemplate(
        input_variables=["motive_list", "event_text", "scores_list_formatted"], 
        template=APPRAISAL_ANALYSIS_TEMPLATE
    )
    chain = prompt | llm_instance
    
    json_string = ""
    try:
        response = chain.invoke(
            {
                "motive_list": motive_list, 
                "event_text": event_text, 
                "scores_list_formatted": scores_list_formatted
            }
        )
        json_string = response.content.strip()
        
        if json_string.startswith("```json"):
            json_string = json_string.lstrip("```json").rstrip("```")
        elif json_string.startswith("```"):
            json_string = json_string.lstrip("```").rstrip("```")

        analysis_data = json.loads(json_string, strict=False)
        return analysis_data
        
    except Exception as e:
        raw_output_snippet = json_string[:200].replace('\n', '\\n')
        st.error(f"Error during LLM Appraisal Analysis. Could not parse JSON. Error: {e}. Raw LLM output: {raw_output_snippet}...")
        return None

# --- Dynamic Interview Logic and Synthesis (Uses first-person 'I') ---
INTERVIEW_PROMPT_TEMPLATE = """
You are a Dynamic Interviewer for a psychological study. Your goal is to collect all 10 key pieces of information (CORE QUESTIONS) about a stressful event from the user's responses, but only ask questions that are relevant or missing.

Your responses must be conversational and contextual.

The user's response history so far is:
{qa_pairs}

The set of ALL 10 CORE QUESTIONS is:
{all_questions}

Your task is:
1. Analyze the Q&A history to determine which CORE QUESTIONS have been sufficiently covered by the user's answers.
2. **CRITICAL RULE:** **Not all 10 CORE QUESTIONS must be explicitly covered.** Use your best judgment to transition to synthesis when the event description feels rich and complete, or if a remaining question is implicitly answered or clearly non-applicable to the specific event.
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

        if json_string.startswith("```json"):
            json_string = json.loads(json_string.lstrip("```json").rstrip("```"), strict=False)
        elif json_string.startswith("```"):
            json_string = json.loads(json_string.lstrip("```").rstrip("```"), strict=False)
        else:
            json_string = json.loads(json_string, strict=False) # Attempt simple load

        result = json_string
        return result
    except Exception as e:
        # Fallback error handling
        st.error(f"Error during LLM Interview Processing. Error: {e}")
        # Return a safe, basic structure to continue the flow
        return {"status": "error", "conversational_response": "I ran into an issue while processing that. Can you please tell me more about what happened?", "next_question": INTERVIEW_QUESTIONS[0], "final_narrative": None}


# --- PROMPT TEMPLATE GENERATION (No change) ---
def get_prompts_for_condition(condition, motive_scores, event_text, analysis_data):
    """Generates the specific system instruction (template) for Guidance."""
    
    # Analysis data now only contains 'congruence_ratings'
    congruence_ratings = analysis_data.get("congruence_ratings", {})

    # This formatted list contains BOTH user Importance (1-7) and LLM Congruence (1-7) scores
    scores_list_formatted = "\n".join([
        f"- {motive} (Importance): {score}/{MOTIVE_SCALE_MAX} | (Congruence): {congruence_ratings.get(motive, 'N/A')}/{MOTIVE_SCALE_MAX}" 
        for motive, score in motive_scores.items()
    ])
    
    # --- 1. Neutral Condition ---
    if condition == "1. Neutral":
        template = """
You are a Baseline Repurposing Assistant. Your task is to generate reappraisal guidance by helping the user identify a motive, value, or goal that the stressful situation they described is, in some way, congruent with. Your guidance must encourage a shift in perspective.

Rules:
1. Do not use any personalization data about the user.
2. The response must be a concise, action-oriented directive focusing on a reframed perspective.
3. Do not repeat the user's story.

User's Event Description: {event_text}
Guidance:
"""
        return PromptTemplate(input_variables=["event_text"], template=template)

    # --- 2. Appraisal-Assessed Condition (Only uses Congruence Scores) ---
    elif condition == "2. Appraisal-Assessed":
        # Only provide the Congruence ratings (1-7 scale) for LLM analysis
        congruence_only_list = "\n".join([
            f"- {motive}: {congruence_ratings.get(motive, 'N/A')}/{MOTIVE_SCALE_MAX}" 
            for motive in motive_scores.keys()
        ])
        
        mock_analysis_data = f"""
The situation analysis (Appraisals) is based on the following congruence scores (1=Low, {MOTIVE_SCALE_MAX}=High):
{congruence_only_list}
"""
        template = f"""
You are an Appraisal-Assessed Repurposing Assistant. Your task is to generate reappraisal guidance. You have access to the situation analysis:
{mock_analysis_data}

Rules:
1. Analyze the congruence scores and **identify all motives that have a high congruence score (5 or higher on the 1-7 scale)**.
2. **FALLBACK:** If no motive meets the 5+ threshold, **identify ALL motives that share the absolute highest congruence score** (the maximum score found in the list).
3. Generate guidance that leverages the identified motive(s) to provide a robust, multifaceted reframe of the event (Repurposing strategy).
4. The response must be a concise, action-oriented directive focusing on a reframed perspective.
5. **CRITICAL:** The final output guidance MUST NOT contain any numbers, scores, or motive ratings (from either the user or the LLM).
6. Do not repeat the user's story.

User's Event Description: {{event_text}}
Guidance:
"""
        return PromptTemplate(input_variables=["event_text"], template=template)

    # --- 3. Appraisal-Aware Condition (Uses Importance (1-7) and Congruence (1-7)) ---
    elif condition == "3. Appraisal-Aware":
        motivational_profile = f"""
The user's full Motive Profile, combining both Importance and Congruence scores, is:
{scores_list_formatted}

(Importance is the user's self-rating on a 1-{MOTIVE_SCALE_MAX} scale; Congruence is the LLM's analysis on a 1-{MOTIVE_SCALE_MAX} scale.)
"""
        template = f"""
You are a Personalized Repurposing Coach. Your task is to generate highly personalized reappraisal guidance. You have access to the user's full motivational profile and situation analysis:
{motivational_profile}

Rules:
1. Analyze the profile to find **all motives** that are **both highly important (user score of 5 or higher) and highly congruent (LLM score of 5 or higher)**. This set represents the most viable targets for repurposing.
2. **FALLBACK:** If no motive meets the combined high threshold, **identify ALL motives that share the absolute highest LLM congruence score** (the maximum congruence score found in the list), regardless of their importance score.
3. Generate guidance that specifically attempts to activate this **set of personalized target motives** or the fallback set to help them re-evaluate the stressful event using the repurposing strategy.
4. The guidance should prioritize framing the situation as an opportunity to reinforce or demonstrate these target motives.
5. The response must be a concise, action-oriented directive.
6. **CRITICAL:** The final output guidance MUST NOT contain any numbers, scores, or motive ratings (from either the user or the LLM).
7. Do not repeat the user's story.

User's Event Description: {{event_text}}
Guidance:
"""
        return PromptTemplate(input_variables=["event_text"], template=template)

    # If no condition matched, return None, causing the TypeError downstream
    return None

# --- 3. DATA SAVING LOGIC (No change) ---
def save_data(data):
    """Saves the comprehensive trial data as a new document in Firestore."""
    try:
        db.collection(COLLECTION_NAME).add(data)
        return True
    except Exception as e:
        st.error(f"❌ Failed to save data: {e}. Check Firestore rules and credentials.")
        return False

# --- 4. PAGE RENDERING FUNCTIONS ---

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
    
    if st.button("I Consent to Participate", type="primary"):
        st.session_state.page = 'motives'
        st.rerun()

def show_motives_page():
    """Renders the Motive Assessment page (1-7 Radio Buttons)."""
    st.title("🎯 Motive Importance Assessment")
    st.markdown(f"""
    Please rate how important each of the following psychological motives is **to you** personally, on a scale of 1 to {MOTIVE_SCALE_MAX}.

    **1 = Not Important At All** | **{MOTIVE_SCALE_MAX} = Extremely Important**
    """)

    motive_scores = {}
    with st.form("motive_form"):
        # Create radio buttons for each motive
        for motive in MOTIVES:
            st.markdown(f"**How important is {motive} to you?**")
            motive_scores[motive] = st.radio(
                label=f"Rating for {motive}",
                options=list(range(1, MOTIVE_SCALE_MAX + 1)),
                index=3, # Default to the middle of the 1-7 scale
                horizontal=True,
                key=f"motive_radio_{motive}"
            )
        
        if st.form_submit_button("Next: Start Interview"): 
            # Ensure all values are collected before proceeding
            if all(motive_scores.values()):
                st.session_state.motive_scores = motive_scores
                st.session_state.page = 'chat' # Navigate to new chat page
                st.rerun()
            else:
                st.warning("Please rate all motives before proceeding.")

def show_chat_page():
    """Renders the guided chat interview to collect event details and synthesize a narrative."""
    st.header("🗣️ Event Interview: Tell Me Your Story")
    st.markdown("This structured conversation ensures we gather all the necessary context about your event for the AI analysis.")

    # Initialize chat state
    if 'interview_messages' not in st.session_state:
        st.session_state.interview_messages = [AIMessage(content="Hello! Let's start by hearing your story: " + INTERVIEW_QUESTIONS[0])]
        st.session_state.interview_answers = []
        st.session_state.event_text_synthesized = None

    messages = st.session_state.interview_messages
    answers = st.session_state.interview_answers
    
    # --- Handle completion and transition ---
    if st.session_state.event_text_synthesized:
        st.success("✅ Interview complete! Story compiled. Proceed to the next stage.")
        if st.button("Next: Review Event Description", type="primary", use_container_width=True):
            st.session_state.page = 'experiment'
            st.rerun()
        return

    # --- Conversation History ---
    st.markdown("#### Conversation History")
    # Fixed height for chat history container
    chat_container = st.container(height=450, border=True)

    with chat_container:
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    # --- Manual Skip Button Logic (Placed above the sticky chat input) ---
    st.markdown("---")
    skip_button_clicked = st.button("Skip Interview & Use Current Story", type="secondary", use_container_width=True)
    
    if skip_button_clicked:
        
        # Check for answers and display error without calling st.rerun yet.
        if not answers:
            st.error("Please provide at least a starting description of the event before skipping.")
        else:
            with st.spinner("Compiling your story..."):
                # Manually trigger the synthesis logic
                interview_result = process_interview_step(llm, answers, is_skip=True)
                
                if interview_result['status'] == 'complete':
                    st.session_state.event_text_synthesized = interview_result['final_narrative']
                    messages.append(AIMessage(content=interview_result['conversational_response']))
                    # Direct transition to the experiment page after successful skip.
                    st.session_state.page = 'experiment' 
                elif interview_result['status'] == 'error':
                     messages.append(AIMessage(content=interview_result['conversational_response']))
            
            # Rerun is placed here to trigger the state update only after processing the skip.
            st.rerun() 

    # --- User Input Loop (Standard sticky chat input) ---
    if user_input := st.chat_input("Your Response:"):
        
        # 1. Store user response
        messages.append(HumanMessage(content=user_input))
        
        # Determine the question the user just answered (last AI message)
        question_just_answered = "Initial Prompt" # Fallback
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                question_just_answered = msg.content
                break
        
        # Store the Q&A pair (only storing the user's input/response for the history)
        answers.append({"question": question_just_answered, "answer": user_input})
        
        # 2. Get LLM to process and determine next step
        with st.spinner("Processing..."): 
            
            interview_result = process_interview_step(llm, answers, is_skip=False)
            
            if interview_result['status'] == 'continue':
                # Add conversational response and next question
                messages.append(AIMessage(content=f"{interview_result['conversational_response']} {interview_result['next_question']}"))
            
            elif interview_result['status'] == 'complete':
                # Interview finished! Store narrative and display conclusion.
                st.session_state.event_text_synthesized = interview_result['final_narrative']
                messages.append(AIMessage(content=interview_result['conversational_response']))
            
            elif interview_result['status'] == 'error':
                 messages.append(AIMessage(content=interview_result['conversational_response']))
                 # If an error, we rely on the error message to guide the user.

        st.rerun() # Rerun to show updated state and messages


def show_experiment_page():
    """Renders the Core Experiment Logic page, generating and collecting ratings for all conditions simultaneously."""
    
    # --- REDIRECTION GUARD (Fixes flicker on submit) ---
    if st.session_state.get('is_redirecting', False):
        return
    # ----------------------------------------------------
    
    st.title("🧪 Experiment: Guidance Elicitation & Comparison")
    
    # Check dependencies
    if 'motive_scores' not in st.session_state or 'event_text_synthesized' not in st.session_state:
        st.warning("Please complete the Motive Assessment and Event Interview first.")
        st.session_state.page = 'motives' if 'motive_scores' not in st.session_state else 'chat'
        return
    
    # Initialize necessary state variables
    if 'is_generating_all' not in st.session_state:
        st.session_state.is_generating_all = False
    if 'event_text_for_llm' not in st.session_state:
        st.session_state.event_text_for_llm = ""
        
    # Pre-populate event input with synthesized text
    if 'event_text_synthesized' in st.session_state and 'event_input' not in st.session_state:
        st.session_state.event_input = st.session_state.event_text_synthesized

    # Event Input Area
    st.subheader("Event Description Review")

    # Text Area Definition: The actual value is stored in st.session_state["event_input"]
    st.text_area(
        "Describe a recent, challenging, or stressful event in detail (Edit the synthesized text if needed):", 
        key="event_input",
        height=200,
        placeholder=f"Example: I have been working 18-hour days to meet a client deadline, and I worry about the quality of my output and missing my child's recital. (Minimum {MIN_EVENT_LENGTH} characters required)",
    )
    
    # Read the current content directly from session state (which holds the latest input)
    event_text = st.session_state.get("event_input", "") 
    current_length = len(event_text)

    # --- BUTTON LOGIC: Only disabled if generation is running or ratings are showing ---
    button_disabled = st.session_state.is_generating_all or st.session_state.get('show_all_ratings', False)
    
    if st.button("Generate All Guidance Responses", type="primary", use_container_width=True, disabled=button_disabled):
        
        # 1. Input Validation Check 
        if current_length < MIN_EVENT_LENGTH:
            remaining_chars = MIN_EVENT_LENGTH - current_length
            st.error(
                f"⚠️ Please ensure your event description is substantial. A minimum of {MIN_EVENT_LENGTH} characters is required for proper LLM analysis. You need **{remaining_chars}** more characters."
            )
            st.session_state.is_generating_all = False 
            return

        # 2. Start Generation Process
        st.session_state.is_generating_all = True
        st.session_state.event_text_for_llm = event_text 
        st.session_state.event_description_edited = event_text 
        st.rerun()

    # --- LLM EXECUTION PHASE ---
    if st.session_state.get('is_generating_all', False):
        
        event_text_to_process = st.session_state.event_text_for_llm
        all_guidance_data = {}
        analysis_data = None

        # Using a spinner while the LLM runs
        with st.spinner("Analyzing Congruence and Generating Guidance for all 3 versions..."):
            
            # 1. Appraisal Analysis (Run ONCE)
            analysis_data = run_appraisal_analysis(llm, st.session_state.motive_scores, event_text_to_process)
            
            if analysis_data:
                # Store analysis data once at the top level of session state
                st.session_state.appraisal_analysis = analysis_data 
                
                # 2. Guidance Generation for EACH condition
                # Use enumerate to pair user-facing label with internal condition name
                for i, condition in enumerate(CONDITION_OPTIONS):
                    user_label = GUIDANCE_LABELS[i]
                    
                    prompt_template = get_prompts_for_condition(
                        condition, st.session_state.motive_scores, event_text_to_process, analysis_data
                    )
                    
                    chain = prompt_template | llm
                    try:
                        response = chain.invoke({"event_text": event_text_to_process})
                        guidance = response.content
                        
                        all_guidance_data[user_label] = {
                            "guidance": guidance,
                            "condition_id": condition, # Store internal condition name for data saving
                        }
                        
                    except Exception as e:
                        st.error(f"An error occurred during LLM Guidance generation for '{condition}'. Error: {e}")
                        all_guidance_data[user_label] = {
                            "guidance": f"ERROR: Could not generate guidance for {condition}.", 
                            "condition_id": condition, 
                        }

                # Success/Completion: Store data and prepare for display
                st.session_state.all_guidance_data = all_guidance_data
                st.session_state.show_all_ratings = True
            
            else:
                st.session_state.show_all_ratings = False
                # If analysis failed, clear the temporary analysis state
                if 'appraisal_analysis' in st.session_state:
                    del st.session_state.appraisal_analysis
                st.error("Failed to run initial Appraisal Analysis. Cannot generate guidance.")
    
        # 3. Clean up and trigger final display rerun
        st.session_state.is_generating_all = False
        if 'event_text_for_llm' in st.session_state:
            del st.session_state.event_text_for_llm
        st.rerun()

    # --- Participant Rating Collection and Data Submission ---
    if st.session_state.get('show_all_ratings', False):
        
        st.markdown("---")
        st.markdown("### Participant Rating & Submission")
        st.write("Please review and rate each guidance message below. For research purposes, **comments are required** for each guidance message.")
        
        # Initialize all collected ratings structure
        if "all_collected_ratings" not in st.session_state:
            st.session_state.all_collected_ratings = {
                label: {dim: 4 for dim in RATING_DIMENSIONS} 
                for label in GUIDANCE_LABELS
            }
        
        # Initialize comment state structure
        if "guidance_comments_by_label" not in st.session_state:
             st.session_state.guidance_comments_by_label = {label: "" for label in GUIDANCE_LABELS}

        # Initialize overall comments field (optional, separate from per-guidance comments)
        if "overall_comments" not in st.session_state:
             st.session_state.overall_comments = ""
        
        
        with st.form("all_ratings_form"):
            
            # --- START 3-COLUMN LAYOUT ---
            # Columns use the full available width of the current *centered* container.
            cols = st.columns(len(GUIDANCE_LABELS)) 
            
            # UI Loop for all three guidance responses
            for i, label in enumerate(GUIDANCE_LABELS):
                with cols[i]: # Place content into the corresponding column
                    guidance = st.session_state.all_guidance_data.get(label, {}).get('guidance', 'Guidance not available.')
                    
                    st.subheader(f"{label}")
                    
                    with st.container(border=True):
                        st.markdown("#### 💬 Generated Guidance")
                        # Using an alert style container for the guidance text
                        st.success(guidance) 
                        
                        # Inner loop for ratings
                        for dim in RATING_DIMENSIONS:
                            current_rating = st.session_state.all_collected_ratings[label].get(dim, 4)
                            
                            st.session_state.all_collected_ratings[label][dim] = st.slider(
                                f"{dim} (1-{MOTIVE_SCALE_MAX})", 
                                1, MOTIVE_SCALE_MAX, 
                                current_rating, 
                                key=f"rating_{label}_{dim}"
                            )

                        # --- INDIVIDUAL GUIDANCE COMMENTS FIELD (MANDATORY CHECK) ---
                        st.session_state.guidance_comments_by_label[label] = st.text_area(
                            f"Required Comments on {label}:", 
                            value=st.session_state.guidance_comments_by_label.get(label, ""),
                            key=f"comments_input_{label}", 
                            height=100, 
                            placeholder="Please provide feedback on this specific guidance message.",
                        )

            # --- END 3-COLUMN LAYOUT ---
            
            # The overall comments and submit button remain full width below the columns
            st.markdown("---")
            st.markdown("### Overall Experience")
            st.session_state.overall_comments = st.text_area(
                "Optional Overall Comments on the experience:", 
                value=st.session_state.overall_comments,
                key="overall_comments_input", 
                height=100, 
                placeholder="Enter any general feedback about the experience or comparison.",
            )

            # Submission button
            submit_button = st.form_submit_button("Submit All Ratings and Save Trial Data", type="primary")

            if submit_button:
                
                # Check for mandatory comments
                missing_comments = [
                    label for label in GUIDANCE_LABELS 
                    if not st.session_state.guidance_comments_by_label.get(label, "").strip()
                ]
                
                if missing_comments:
                    st.error(f"⚠️ **Please provide comments for all guidance messages.** Missing comments for: {', '.join(missing_comments)}")
                    # Do NOT proceed with submission
                    return

                # Consolidate results for saving
                results_by_condition = {}
                for label in GUIDANCE_LABELS:
                    data = st.session_state.all_guidance_data[label]
                    
                    results_by_condition[data['condition_id']] = {
                        "guidance_label": label, # Store the user-facing label for clarity
                        "llm_guidance": data['guidance'],
                        "participant_ratings": st.session_state.all_collected_ratings[label],
                        "participant_comments_on_guidance": st.session_state.guidance_comments_by_label[label],
                    }

                # Prepare final data for Firestore
                trial_data = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "motive_importance_scores": st.session_state.motive_scores, 
                    "synthesized_event_narrative": st.session_state.event_text_synthesized,
                    "event_description_edited": st.session_state.event_description_edited,
                    "interview_qa_history": st.session_state.interview_answers, 
                    "overall_participant_comments": st.session_state.overall_comments, 
                    # Store the single set of appraisal analysis data from the root state
                    "appraisal_analysis": st.session_state.appraisal_analysis,
                    "results_by_condition": results_by_condition, 
                }
                
                if save_data(trial_data):
                    # Reset state and trigger page change
                    st.session_state.show_all_ratings = False
                    st.session_state.is_redirecting = True 
                    st.session_state.page = 'thank_you' 
                    st.rerun()

def show_thank_you_page():
    """Renders the Thank You page with option to restart the experiment."""
    st.title("🎉 Thank You for Participating!")
    st.success("Your trial data has been successfully submitted and saved.")

    st.markdown("""
    Your contribution is valuable to our research on personalized cognitive strategies.

    Would you like to run the experiment one more time with a **different stressful event**?
    *(Note: Your current Motive Importance Assessments will be used for the next trial.))*
    """)

    if st.button("Run Another Trial", type="primary"):
        # Reset the trial-specific data and redirect flag, ensuring the experiment restarts cleanly.
        for key in ['all_guidance_data', 'all_collected_ratings', 'show_all_ratings', 'event_description_edited', 'event_input', 'is_redirecting', 'is_generating_all', 'guidance_comments_by_label', 'overall_comments', 'interview_messages', 'interview_answers', 'event_text_synthesized', 'appraisal_analysis']: 
            if key in st.session_state:
                del st.session_state[key]
        
        # Go directly to the chat page to start the next trial
        st.session_state.page = 'chat' 
        st.rerun()


# --- 5. MAIN APP EXECUTION ---

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = 'consent'


if st.session_state.page == 'consent':
    show_consent_page()
elif st.session_state.page == 'motives':
    show_motives_page()
elif st.session_state.page == 'chat': 
    show_chat_page()
elif st.session_state.page == 'experiment':
    show_experiment_page()
elif st.session_state.page == 'thank_you': 
    show_thank_you_page()
