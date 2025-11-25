import streamlit as st
import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage # Added for chat functionality

# --- 1. CONFIGURATION & SETUP ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.5 
MOTIVES = ["Achievement & Success", "Security & Stability", "Affiliation & Belonging", "Stimulation & Excitement", "Self-Direction & Autonomy"]
RATING_DIMENSIONS = ["Believability", "Appropriateness", "Emotional Valence"]
MOTIVE_SCALE_MAX = 7 # All ratings will be on a 1-7 scale
MIN_EVENT_LENGTH = 200 # Minimum number of characters required for the event description

INTERVIEW_QUESTIONS = [
    "What happened? Describe this as if you were telling the story to a friend.",
    "Who else was around or involved?",
    "What else should we know about the situation? What was the context?",
    "What was the outcome?",
    "Why did this situation matter to you? What made it important?",
    "Some situations are predictable and clear, others less so. How about this one, was it predictable in any way? Why?",
    "Who could be considered responsible for this situation happening?",
    "Would you say you were in control of this situation? Why?",
    "What would you have wanted to happen in this situation, if you could change anything?",
    "How big of a deal this discrepancy is for you?",
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
        st.sidebar.success("✅ LLM Initialized")
        return llm
    except Exception as e:
        st.sidebar.error(f"❌ LLM Initialization Failed: {e}")
        st.stop()

llm = get_llm()


# --- 2. LLM LOGIC FUNCTIONS ---

# --- LLM APPRAISAL ANALYSIS ---
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

# --- LLM Synthesis Function ---
SYNTHESIS_TEMPLATE = """
You are a Narrative Synthesizer. Your task is to take a set of interview questions and the user's answers, provided below, and combine them into a single, cohesive, narrative description of the event.

The output must be a unified story, suitable for being the main event description in a psychological study. Do not include the questions or any introductory/concluding text. Just output the narrative.

Q&A Pairs:
{qa_pairs}

Cohesive Event Narrative:
"""

@st.cache_data(show_spinner=False)
def synthesize_event_text(llm_instance, qa_pairs_formatted):
    """Executes the LLM to generate a single narrative from Q&A pairs."""
    prompt = PromptTemplate(
        input_variables=["qa_pairs"], 
        template=SYNTHESIS_TEMPLATE
    )
    chain = prompt | llm_instance
    
    try:
        response = chain.invoke({"qa_pairs": qa_pairs_formatted})
        return response.content.strip()
    except Exception as e:
        st.error(f"Error during LLM Synthesis: {e}")
        return None


# --- PROMPT TEMPLATE GENERATION ---
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

# --- 3. DATA SAVING LOGIC ---
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
        
        if st.form_submit_button("Next: Start Interview"): # Updated button text
            # Ensure all values are collected before proceeding
            if all(motive_scores.values()):
                st.session_state.motive_scores = motive_scores
                st.session_state.page = 'chat' # Navigate to new chat page
                st.rerun()
            else:
                st.warning("Please rate all motives before proceeding.")

def show_chat_page():
    """Renders the guided chat interview to collect event details and synthesize a narrative."""
    st.title("🗣️ Event Interview")
    st.markdown("Before proceeding, please answer the questions below to help us create a detailed event description.")

    # Initialize chat state
    if 'interview_messages' not in st.session_state:
        st.session_state.interview_messages = [AIMessage(content="Hello! Let's start with the first question: " + INTERVIEW_QUESTIONS[0])]
        st.session_state.interview_answers = []
        st.session_state.current_q_index = 0
        st.session_state.event_text_synthesized = None

    messages = st.session_state.interview_messages
    answers = st.session_state.interview_answers
    q_index = st.session_state.current_q_index
    
    # --- Display chat history ---
    chat_container = st.container(height=400, border=True)

    with chat_container:
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    # --- Handle input and progression ---
    if st.session_state.event_text_synthesized:
        st.success("✅ Interview complete! Click Next to review the synthesized event description.")
        if st.button("Next: Review Event Description", type="primary", use_container_width=True):
            st.session_state.page = 'experiment'
            st.rerun()
        return

    # User input is only accepted if there are still questions to ask
    if q_index < len(INTERVIEW_QUESTIONS):
        
        # Determine the prompt text for the input box
        input_prompt = f"Answer for Question {q_index + 1}/{len(INTERVIEW_QUESTIONS)}: {INTERVIEW_QUESTIONS[q_index]}"
        
        if user_input := st.chat_input(input_prompt):
            
            # 1. Store user response and display
            messages.append(HumanMessage(content=user_input))
            answers.append({"question": INTERVIEW_QUESTIONS[q_index], "answer": user_input})
            
            # 2. Advance question index
            st.session_state.current_q_index += 1
            
            # 3. Check if interview is finished
            if st.session_state.current_q_index == len(INTERVIEW_QUESTIONS):
                # Interview finished: Synthesize the final event text
                with st.spinner("Synthesizing final event narrative from your answers..."):
                    
                    qa_pairs_formatted = "\n---\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answers])
                    
                    synthesized_text = synthesize_event_text(llm, qa_pairs_formatted)
                    
                    if synthesized_text:
                        st.session_state.event_text_synthesized = synthesized_text
                        st.session_state.messages.append(AIMessage(content="Thank you for the detailed answers! The full narrative has been synthesized. Click 'Next' below."))
                    else:
                        st.session_state.messages.append(AIMessage(content="I encountered an error while summarizing. Please try again."))
                        st.session_state.current_q_index -= 1 # Revert index
                        
            else:
                 # Ask the next question
                 next_q = INTERVIEW_QUESTIONS[st.session_state.current_q_index]
                 messages.append(AIMessage(content=f"Understood. Moving on to question {st.session_state.current_q_index + 1}: {next_q}"))
            
            st.rerun() # Rerun to show updated state and messages


def show_experiment_page():
    """Renders the Core Experiment Logic page."""
    
    # --- REDIRECTION GUARD (Fixes flicker on submit) ---
    if st.session_state.get('is_redirecting', False):
        return
    # ----------------------------------------------------
    
    st.title("🧪 Experiment: Event Elicitation & Guidance")
    
    # Check if motive scores are available
    if 'motive_scores' not in st.session_state:
        st.warning("Please complete the Motive Assessment first.")
        st.session_state.page = 'motives'
        return
    
    # Check if event text was synthesized (i.e., chat page was completed)
    if 'event_text_synthesized' not in st.session_state:
        st.warning("Please complete the Event Interview first.")
        st.session_state.page = 'chat'
        return
    
    # Initialize necessary state variables
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False
    if 'event_text_for_llm' not in st.session_state:
        st.session_state.event_text_for_llm = ""
        
    # --- New: Pre-populate event input with synthesized text ---
    if 'event_text_synthesized' in st.session_state and 'event_input' not in st.session_state:
        # Only pre-populate if the user has completed the interview and hasn't started editing this field yet
        st.session_state.event_input = st.session_state.event_text_synthesized

    # Experiment Condition Selection (Radio buttons on top)
    condition_options = ["1. Neutral", "2. Appraisal-Assessed", "3. Appraisal-Aware"]
    default_index = condition_options.index("1. Neutral") 
    
    selected_condition = st.radio(
        "Select Experimental Condition:",
        condition_options,
        index=default_index, 
        horizontal=True
    )
    # Store selected condition immediately for the LLM execution phase
    st.session_state.selected_condition = selected_condition 

    # Event Input Area
    st.subheader(f"Condition Selected: {selected_condition}")

    # Text Area Definition: The actual value is stored in st.session_state["event_input"]
    st.text_area(
        "Describe a recent, challenging, or stressful event in detail (Edit the synthesized text if needed):", # Updated prompt
        key="event_input",
        height=200,
        placeholder=f"Example: I have been working 18-hour days to meet a client deadline, and I worry about the quality of my output and missing my child's recital. (Minimum {MIN_EVENT_LENGTH} characters required)",
    )
    
    # Read the current content directly from session state (which holds the latest input)
    event_text = st.session_state.get("event_input", "") 
    current_length = len(event_text)

    # --- BUTTON LOGIC: Only disabled if generation is running ---
    # The button is now always clickable unless actively generating.
    button_disabled = st.session_state.is_generating
    
    if st.button("Generate Repurposing Guidance", type="primary", use_container_width=True, disabled=button_disabled):
        
        # 1. Input Validation Check (runs AFTER button is pressed)
        if current_length < MIN_EVENT_LENGTH:
            remaining_chars = MIN_EVENT_LENGTH - current_length
            # Display a temporary error message that shows the required length
            st.error(
                f"⚠️ Please ensure your event description is substantial. A minimum of {MIN_EVENT_LENGTH} characters is required for proper LLM analysis. You need **{remaining_chars}** more characters."
            )
            
            # Reset state and return to prevent LLM call
            st.session_state.event_text_for_llm = ""
            st.session_state.is_generating = False 
            return # Stop execution and wait for the user to fix the input

        # 2. Start Generation Process
        st.session_state.is_generating = True
        st.session_state.event_text_for_llm = event_text # Use the validated text
        st.rerun()

    # --- LLM EXECUTION PHASE ---
    if st.session_state.get('is_generating', False) and 'event_text_for_llm' in st.session_state:
        
        event_text_to_process = st.session_state.event_text_for_llm
        analysis_data = None
        guidance = ""

        # Using a spinner while the LLM runs
        with st.spinner("Analyzing Congruence and Generating Guidance..."):
            
            # 1. Appraisal Analysis
            # The result is now only the {'congruence_ratings': {...}} object
            analysis_data = run_appraisal_analysis(llm, st.session_state.motive_scores, event_text_to_process)
            
            if analysis_data:
                
                # 2. Guidance Generation
                prompt_template = get_prompts_for_condition(
                    st.session_state.selected_condition, st.session_state.motive_scores, event_text_to_process, analysis_data
                )
                
                chain = prompt_template | llm
                try:
                    response = chain.invoke({"event_text": event_text_to_process})
                    guidance = response.content
                    
                    # Success: Store data and prepare for display
                    st.session_state.final_guidance = guidance
                    st.session_state.analysis_data = analysis_data
                    st.session_state.event_text = event_text_to_process
                    st.session_state.show_ratings = True
                    
                except Exception as e:
                    st.error(f"An error occurred during LLM Guidance generation. Error: {e}")
                    st.session_state.show_ratings = False
            else:
                st.session_state.show_ratings = False
    
        # 3. Clean up and trigger final display rerun
        st.session_state.is_generating = False
        if 'event_text_for_llm' in st.session_state:
            del st.session_state.event_text_for_llm
        st.rerun()

    # --- Participant Rating Collection and Data Submission ---
    if st.session_state.get('show_ratings', False) and 'final_guidance' in st.session_state:
        
        st.markdown("---")
        st.markdown("### Participant Rating & Submission")
        st.write("Rate the generated guidance based on your experience:")
        
        # Using a container to wrap the guidance
        with st.container(border=True):
            st.markdown("#### 💬 Generated Guidance")
            st.success(st.session_state.final_guidance)

        # Initialize ratings in session state if not present
        if "collected_ratings" not in st.session_state:
            st.session_state.collected_ratings = {dim: 4 for dim in RATING_DIMENSIONS}
        
        # Initialize comment state if not present
        if "guidance_comments" not in st.session_state:
             st.session_state.guidance_comments = ""
        
        with st.form("rating_form"):
            
            # Rating Sliders - Uses MOTIVE_SCALE_MAX (7)
            for dim in RATING_DIMENSIONS:
                # Use value from session state if it exists, otherwise default to 4
                default_value = st.session_state.collected_ratings.get(dim, 4)
                st.session_state.collected_ratings[dim] = st.slider(
                    f"{dim} (1-{MOTIVE_SCALE_MAX})", 
                    1, MOTIVE_SCALE_MAX, default_value, 
                    key=dim
                )
            
            # --- COMMENTS FIELD ---
            st.session_state.guidance_comments = st.text_area(
                "Optional Comments on Guidance:", 
                key="comments_input", 
                height=100, 
                placeholder="Enter any feedback, thoughts, or suggestions about the guidance here."
            )
            
            # Submission button
            if st.form_submit_button("Submit Ratings and Save Trial Data"):
                
                # Prepare data for Firestore
                trial_data = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "condition": st.session_state.selected_condition,
                    "event_description": st.session_state.event_text,
                    "motive_importance_scores": st.session_state.motive_scores, # User's 1-7 ratings
                    "appraisal_analysis": st.session_state.analysis_data, # LLM's structured analysis (now just congruence_ratings)
                    "llm_guidance": st.session_state.final_guidance,
                    "participant_ratings": st.session_state.collected_ratings, # User's 1-7 ratings for guidance
                    "participant_comments": st.session_state.guidance_comments, # <--- NEW FIELD
                }
                
                if save_data(trial_data):
                    # Hide ratings, set redirect flag, and trigger page change
                    st.session_state.show_ratings = False
                    st.session_state.is_redirecting = True # <--- Flicker Fix
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
        for key in ['final_guidance', 'analysis_data', 'selected_condition', 'event_text', 'collected_ratings', 'show_ratings', 'event_input', 'is_redirecting', 'is_generating', 'guidance_comments', 'interview_messages', 'interview_answers', 'current_q_index', 'event_text_synthesized']: # Added chat state keys
            if key in st.session_state:
                del st.session_state[key]
        
        # Go directly to the experiment page, using the existing motive scores
        st.session_state.page = 'experiment'
        st.rerun()


# --- 5. MAIN APP EXECUTION ---

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = 'consent'

# Display current page
st.sidebar.title("Protocol Status")
st.sidebar.markdown(f"**Current Page:** `{st.session_state.page}`")

if st.session_state.page == 'consent':
    show_consent_page()
elif st.session_state.page == 'motives':
    show_motives_page()
elif st.session_state.page == 'chat': # New chat page
    show_chat_page()
elif st.session_state.page == 'experiment':
    show_experiment_page()
elif st.session_state.page == 'thank_you': 
    show_thank_you_page()

st.sidebar.markdown("---")
st.sidebar.header("Debugging Data")
st.sidebar.json(st.session_state.get('motive_scores', {'Status': 'Not Collected'}))
