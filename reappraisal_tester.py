import streamlit as st
import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION & SETUP ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.5 
MOTIVES = ["Achievement & Success", "Security & Stability", "Affiliation & Belonging", "Stimulation & Excitement", "Self-Direction & Autonomy"]
RATING_DIMENSIONS = ["Believability", "Appropriateness", "Emotional Valence"]
MOTIVE_SCALE_MAX = 7

try:
    from google.cloud import firestore
except ImportError:
    st.error("The 'google-cloud-firestore' package is not installed. Please check requirements.txt.")
    st.stop()

# 1.1 Securely load Firebase credentials and initialize Firestore client
@st.cache_resource
def get_firestore_client():
    if "gcp_service_account" not in st.secrets:
        st.error("Database Error: 'gcp_service_account' secret not found. Check Streamlit Secrets configuration.")
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
APPRAISAL_ANALYSIS_TEMPLATE = """
You are an Appraisal Analyst. Your task is to analyze the user's event description in the context of their core motives.
Your output MUST be a valid JSON object. Do not include any text, headers, or markdown formatting outside of the JSON block.

The JSON object MUST contain:
"congruence_ratings": A dictionary where keys are the motive names from the MOTIVE LIST and values are a score from 1 (Low Congruence) to 7 (High Congruence).

MOTIVE LIST: {motive_list}
User's Event Description: {event_text}
User's Motive Importance Scores: {scores_list_formatted}

Provide the JSON output:
"""

@st.cache_data(show_spinner=False)
def run_appraisal_analysis(llm_instance, motive_scores, event_text):

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

# --- PROMPT TEMPLATE GENERATION ---
def get_prompts_for_condition(condition, motive_scores, event_text, analysis_data):
    """Generates the specific system instruction (template) for Guidance."""
    
    congruence_ratings = analysis_data.get("congruence_ratings", {})
    scores_list_formatted = motive_scores.values()
    # --- 1. Control Condition ---
    if condition == "1. Neutral":
        template = """
                    You are a Neutral Repurposing Assistant. Your task is to generate reappraisal guidance by helping the user identify a motive, value, or goal that the stressful situation they described is, in some way, congruent with. Your guidance must encourage a shift in perspective.

                    Rules:
                    1. Do not use any personalization data about the user.
                    2. The response must be a concise, action-oriented directive focusing on a reframed perspective.
                    3. Do not repeat the user's story.

                    User's Event Description: {event_text}
                    Guidance:
                    """
        return PromptTemplate(input_variables=["event_text"], template=template)

    # --- 2. Appraisal-Assessed Condition ---
    elif condition == "2. Appraisal-Assessed":

        template = f"""
        You are an Appraisal-Assessed Repurposing Assistant. Your task is to generate reappraisal guidance based on **{congruence_ratings}**.

        Rules:
        1. Generate guidance that leverages the 'Potential congruence' motive to reframe the event.
        2. The response must be a concise, action-oriented directive focusing on a reframed perspective.
        3. Do not repeat the user's story.

        User's Event Description: {{event_text}}
        Guidance:
        """
        return PromptTemplate(input_variables=["event_text"], template=template)

    # --- 3. Appraisal-Aware Condition ---
    elif condition == "3. Appraisal-Aware":
        
        template = f"""
        You are a Personalized Appraisal-Aware Repurposing Coach. Your task is to generate a personalized reappraisal guidance. You have access to both LLM-assessed {congruence_ratings} ratings well as the user's original ratings {scores_list_formatted}. 

        Rules:
        1. Generate guidance that help them re-evaluate the stressful event using the LLM-assessed and original user's ratings.
        2. The response must be a concise, action-oriented directive.
        3. Do not repeat the user's story.

        User's Event Description: {{event_text}}
        Guidance:
        """
        return PromptTemplate(input_variables=["event_text"], template=template)

    return None

# --- 3. DATA SAVING LOGIC ---
def save_data(data):
    """Saves the comprehensive trial data as a new document in Firestore."""
    try:
        db.collection(COLLECTION_NAME).add(data)
        st.success("✅ Trial data saved successfully to Firestore!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save data: {e}. Check Firestore rules and credentials.")
        return False

# --- 4. PAGE RENDERING FUNCTIONS ---

def show_consent_page():
    """Renders Page 1: Letter of Consent."""
    st.title("📄 Step 1: Letter of Consent")
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
    """Renders Page 2: Motive Assessment (1-7 Radio Buttons)."""
    st.title("🎯 Step 2: Motive Importance Assessment")
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
        
        if st.form_submit_button("Next: Start Experiment"):
            # Ensure all values are collected before proceeding
            if all(motive_scores.values()):
                st.session_state.motive_scores = motive_scores
                st.session_state.page = 'experiment'
                st.experimental_rerun()
            else:
                st.warning("Please rate all motives before proceeding.")

def show_experiment_page():
    """Renders Page 3: Core Experiment Logic."""
    
    st.title("🧪 Step 3: Guidance")
    
    # Check if motive scores are available
    if 'motive_scores' not in st.session_state:
        st.warning("Please complete the Motive Assessment first.")
        st.session_state.page = 'motives'
        return
    
    # 5.1 Experiment Condition Selection (Radio buttons on top)
    selected_condition = st.radio(
        "Select Experimental Condition:",
        ["1. Neutral", "2. Appraisal-Assessed", "3. Appraisal-Aware"],
        index=2, # Default to the most personalized condition
        horizontal=True
    )

    # 5.2 Event Input Area
    st.subheader(f"Condition Selected: {selected_condition}")

    event_text = st.text_area(
        "Describe a recent, challenging, or stressful event in detail:",
        key="event_input",
        height=200,
        placeholder="Example: I have been working 18-hour days to meet a client deadline, and I worry about the quality of my output and missing my child's recital.",
    )

    if st.button("Generate Repurposing Guidance", type="primary", use_container_width=True) and event_text:
        
        # --- STAGE 1: LLM Appraisal Analysis (Step 3) ---
        analysis_data = None
        with st.spinner("STAGE 1/2: Analyzing Congruence (Step 3)..."):
            analysis_data = run_appraisal_analysis(llm, st.session_state.motive_scores, event_text)
            
        if analysis_data:
            
            # 1. Get the correct prompt template (Step 4 preparation)
            prompt_template = get_prompts_for_condition(
                selected_condition, st.session_state.motive_scores, event_text, analysis_data
            )
            
            # --- STAGE 2: Guidance Generation (Step 4 - LLM Call 2) ---
            guidance = ""
            with st.spinner(f"STAGE 2/2: Generating Guidance for {selected_condition}..."):
                
                chain = prompt_template | llm
                
                try:
                    response = chain.invoke({"event_text": event_text})
                    guidance = response.content
                    
                    # Store data in session state for later submission
                    st.session_state.final_guidance = guidance
                    st.session_state.analysis_data = analysis_data
                    st.session_state.selected_condition = selected_condition
                    st.session_state.event_text = event_text
                    st.session_state.show_ratings = True
                    
                except Exception as e:
                    st.error(f"An error occurred during LLM Guidance generation. Error: {e}")
                    st.session_state.show_ratings = False
    
    # --- Step 5: Participant Rating Collection and Data Submission ---
    if st.session_state.get('show_ratings', False) and 'final_guidance' in st.session_state:
        
        st.markdown("---")
        st.markdown("### Step 4: Participant Rating & Submission")
        st.write("Rate the generated guidance based on your experience:")
        
        st.markdown("#### 💬 Generated Guidance")
        st.success(st.session_state.final_guidance)

        # Initialize ratings in session state if not present
        if "collected_ratings" not in st.session_state:
            # Initialize with default value for 1-7 scale (usually 4)
            st.session_state.collected_ratings = {dim: 4 for dim in RATING_DIMENSIONS}
        
        with st.form("rating_form"):
            
            # Rating Sliders
            for dim in RATING_DIMENSIONS:
                st.session_state.collected_ratings[dim] = st.slider(
                    f"{dim} (1-{MOTIVE_SCALE_MAX})", 
                    1, MOTIVE_SCALE_MAX, 4, 
                    key=dim
                )
            
            # Submission button
            if st.form_submit_button("Submit Ratings and Save Trial Data"):
                
                # Prepare data for Firestore (Step 6)
                trial_data = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "condition": st.session_state.selected_condition,
                    "event_description": st.session_state.event_text,
                    "motive_importance_scores": st.session_state.motive_scores, # User's 1-7 ratings
                    "appraisal_analysis": st.session_state.analysis_data, # LLM's structured analysis
                    "llm_guidance": st.session_state.final_guidance,
                    "participant_ratings": st.session_state.collected_ratings, # User's 1-7 ratings for guidance
                }
                
                if save_data(trial_data):
                    # Clear state for a new trial and go back to motives or consent
                    del st.session_state.show_ratings
                    st.session_state.page = 'motives' # Loop back to motives page
                    st.success("Data saved. Starting a new trial.")
                    st.experimental_rerun()


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
elif st.session_state.page == 'experiment':
    show_experiment_page()

st.sidebar.markdown("---")
st.sidebar.header("Debugging Data")
st.sidebar.json(st.session_state.get('motive_scores', {'Status': 'Not Collected'}))
