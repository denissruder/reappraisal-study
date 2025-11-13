import streamlit as st
import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- NEW: Firestore Imports (Requires 'google-cloud-firestore' package) ---
try:
    from google.cloud import firestore
except ImportError:
    st.error("The 'google-cloud-firestore' package is not installed. Please check requirements.txt.")
    st.stop()

# --- 1. CONFIGURATION ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.5 # A moderate temperature to allow for creative reframing
MOTIVES = ["Achievement & Success", "Security & Stability", "Affiliation & Belonging", "Stimulation & Excitement", "Self-Direction & Autonomy"]
RATING_DIMENSIONS = ["Believability", "Appropriateness", "Emotional Valence (1=Negative, 5=Positive)"]


# 1.1 Securely load Firebase credentials and initialize Firestore client
@st.cache_resource
def get_firestore_client():
    if "gcp_service_account" not in st.secrets:
        st.error("Database Error: 'gcp_service_account' secret not found. Check Streamlit Secrets configuration.")
        st.stop()
        
    try:
        # Load the dictionary from the single-line string secret, using strict=False to handle hidden characters
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
        st.error("LLM Error: 'GEMINI_API_KEY' secret not found. Please check Streamlit Secrets configuration.")
        st.stop()

    # Set the key as an environment variable for LangChain to pick up
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    
    # Initialize the LLM with configured settings
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)
        st.sidebar.success("✅ LLM Initialized")
        return llm
    except Exception as e:
        st.sidebar.error(f"❌ LLM Initialization Failed: {e}")
        st.stop()

llm = get_llm()


# --- LLM APPRAISAL ANALYSIS (STEP 3) ---

APPRAISAL_ANALYSIS_TEMPLATE = """
You are an Appraisal Analyst. Your task is to analyze the user's event description in the context of their core motives.
Your output MUST be a valid JSON object. Do not include any text, headers, or markdown formatting outside of the JSON block.

The JSON object MUST contain three fields:
1. "conflict_motive": The name of the motive most CONFLICTED by the event from the MOTIVE LIST.
2. "congruence_motive": The name of a motive that shows potential CONGRUENCE (or opportunity for Repurposing) from the MOTIVE LIST.
3. "congruence_ratings": A dictionary where keys are the motive names from the MOTIVE LIST and values are a score from 1 (Low Congruence) to 5 (High Congruence).

MOTIVE LIST: {motive_list}
User's Event Description: {event_text}
User's Motive Importance Scores: {scores_list_formatted}

Provide the JSON output:
"""

@st.cache_data(show_spinner=False)
def run_appraisal_analysis(llm_instance, motive_scores, event_text):
    """Executes the LLM to perform Step 3: Appraisal Analysis using LCEL."""
    
    # Format scores and motive list for injection
    scores_list_formatted = "\n".join([f"- {motive}: {score}/5" for motive, score in motive_scores.items()])
    motive_list = ", ".join(MOTIVES)

    prompt = PromptTemplate(
        input_variables=["motive_list", "event_text", "scores_list_formatted"], 
        template=APPRAISAL_ANALYSIS_TEMPLATE
    )
    
    # LCEL Chain: Prompt | LLM
    chain = prompt | llm_instance
    
    json_string = ""
    try:
        # Run the LLM to get the structured analysis using .invoke()
        response = chain.invoke(
            {
                "motive_list": motive_list, 
                "event_text": event_text, 
                "scores_list_formatted": scores_list_formatted
            }
        )
        
        # Extract and clean up the JSON response
        json_string = response.content.strip()
        
        # Use simple string manipulation to clean the JSON block
        if json_string.startswith("```json"):
            json_string = json_string.lstrip("```json").rstrip("```")
        elif json_string.startswith("```"):
            json_string = json_string.lstrip("```").rstrip("```")

        analysis_data = json.loads(json_string, strict=False)
        return analysis_data
        
    except Exception as e:
        # Store the problematic string for error reporting
        raw_output_snippet = json_string[:200].replace('\n', '\\n')
        st.error(f"Error during LLM Appraisal Analysis (Step 3). Could not parse JSON. Error: {e}. Raw LLM output: {raw_output_snippet}...")
        return None


# --- PROMPT TEMPLATE GENERATION (STEP 4) ---

def get_prompts_for_condition(condition, motive_scores, event_text, analysis_data):
    """
    Generates the specific system instruction (template) for Guidance (Step 4)
    using the LLM-generated analysis_data (Step 3).
    """
    
    # Extract data from LLM-generated analysis (Step 3)
    mock_conflict = analysis_data.get("conflict_motive", "N/A")
    mock_congruence = analysis_data.get("congruence_motive", "N/A")
    congruence_ratings = analysis_data.get("congruence_ratings", {})

    # --- Data formatting for Condition 3 ---
    highest_score = max(motive_scores.values())
    most_important_motives = [m for m, s in motive_scores.items() if s == highest_score]
    core_motive = most_important_motives[0] 

    # Explicitly convert congruence ratings to string for formatting
    scores_list_formatted = "\n".join([
        f"- {motive} (Importance): {score}/5 | (Congruence): {congruence_ratings.get(motive, 'N/A')}/5" 
        for motive, score in motive_scores.items()
    ])
    
    # --- 1. Control Condition ---
    if condition == "1. Control":
        template = """
You are a Baseline Repurposing Assistant. Your task is to generate reappraisal guidance by helping the user identify a motive, value, or goal that the stressful situation they described is, in some way, congruent with. Your guidance must encourage a shift in perspective.

Rules:
1. Do not use any personalization data about the user.
2. The response must be a concise, action-oriented directive focusing on a reframed perspective.
3. Do not repeat the user's story.

User's Event Description: {event_text}
Guidance:
"""
        return PromptTemplate(input_variables=["event_text"], template=template), None, congruence_ratings

    # --- 2. Appraisal-Aware Condition ---
    elif condition == "2. Appraisal-Aware":
        # Inject only the Appraisal Analysis data (Step 3)
        mock_analysis_data = f"""
The user's event has been analyzed based on potential motives (Appraisals - Step 3):
- High conflict detected with motive: '{mock_conflict}'.
- Potential congruence (Repurposing target) found with motive: '{mock_congruence}'.
- Congruence Ratings: {json.dumps(congruence_ratings, indent=2)}
"""
        template = f"""
You are an Appraisal-Aware Repurposing Assistant. Your task is to generate reappraisal guidance. You have access to the situation analysis:
{mock_analysis_data}

Rules:
1. Generate guidance that leverages the 'Potential congruence' motive ('{mock_congruence}') to reframe the event.
2. The response must be a concise, action-oriented directive focusing on a reframed perspective.
3. Do not repeat the user's story.

User's Event Description: {{event_text}}
Guidance:
"""
        return PromptTemplate(input_variables=["event_text"], template=template), mock_analysis_data, congruence_ratings

    # --- 3. Importance and Appraisal Aware Condition ---
    elif condition == "3. Importance and Appraisal Aware":
        # Inject both Motive Importance (Step 1) and Appraisal Analysis (Step 3)
        motivational_profile = f"""
The user's full Motive Importance Profile (Step 1) is:
{scores_list_formatted}

The user's **PRIMARY CORE MOTIVE** (highest score) is: **{core_motive}**.
The situation analysis (Step 3) shows high conflict with motive: '{mock_conflict}'.
The situation analysis (Step 3) shows potential congruence with motive: '{mock_congruence}'.
"""
        template = f"""
You are a Personalized Repurposing Coach. Your task is to generate highly personalized reappraisal guidance. You have access to the user's full motivational profile and situation analysis:
{motivational_profile}

Rules:
1. Generate guidance that specifically attempts to **activate the user's PRIMARY CORE MOTIVE: '{core_motive}'** to help them re-evaluate the stressful event using the repurposing strategy.
2. The guidance should prioritize framing the situation as an opportunity to reinforce or demonstrate this core motive.
3. The response must be a concise, action-oriented directive.
4. Do not repeat the user's story.

User's Event Description: {{event_text}}
Guidance:
"""
        return PromptTemplate(input_variables=["event_text"], template=template), motivational_profile, congruence_ratings

    return None, None, {}


# --- DATA SAVING LOGIC (Step 6 - Refined) ---

def save_data(data):
    """Saves the comprehensive trial data as a new document in Firestore."""
    try:
        db.collection(COLLECTION_NAME).add(data)
        st.success("✅ Trial data saved successfully to Firestore!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save data: {e}. Check Firestore rules and credentials.")
        return False


# --- STREAMLIT UI and EXECUTION ---

st.set_page_config(layout="wide", page_title="Reappraisal Experiment Tester")
st.title("🔬 Repurposing Reappraisal Experiment Tester")
st.caption("Simulates the three experimental conditions for personalized guidance based on motivational profiles.")

# --- Sidebar for Controls (Step 1) ---
motive_scores = {}
with st.sidebar:
    st.header("Experiment Controls")
    
    selected_condition = st.radio(
        "Select Experimental Condition:",
        ["1. Control", "2. Appraisal-Aware", "3. Importance and Appraisal Aware"],
        index=2 # Default to the fully personalized condition for demonstration
    )
    
    st.subheader("Step 1: Motive Importance")
    st.write("Rate each motive on a scale of 1 to 5:")
    
    # Simulates Step 1: Motive Importance Measurement using sliders
    for motive in MOTIVES:
        motive_scores[motive] = st.slider(f"{motive}", 1, 5, 3, key=f"motive_score_{motive}") 
    
    st.markdown("---")


# --- Main Content Area (Step 2 - Event Input) ---

st.subheader("Step 2: Event Elicitation (User Input)")

event_text = st.text_area(
    "Describe a recent, challenging, or stressful event in detail:",
    key="event_input",
    height=200,
    placeholder="Example: I have been working 18-hour days to meet a client deadline, and I worry about the quality of my output and missing my child's recital.",
    value="After a sudden merger, my department is being completely overhauled, and a new, much younger VP has been appointed. I have a 30-day ultimatum to develop an unfamiliar system or face termination. This coincides with a major financial commitment for my child's university tuition. I am working 18-hour days, feeling immense pressure, and my health is suffering due to the toxic environment.",
)

if st.button("Generate Repurposing Guidance", type="primary", use_container_width=True) and event_text:
    
    # --- STAGE 1: LLM Appraisal Analysis (Step 3) ---
    analysis_data = None
    with st.spinner("STAGE 1/2: Analyzing Congruence (Step 3 - LLM Call 1)..."):
        # The appraisal analysis is performed here
        analysis_data = run_appraisal_analysis(llm, motive_scores, event_text)
        
    if analysis_data:
        
        st.subheader(f"Guidance Generation: {selected_condition}")

        # Display the Congruence Ratings
        with st.expander("Show LLM-Generated Congruence Ratings (Step 3 Output)"):
            st.markdown(f"**Motive Most Conflicted:** `{analysis_data.get('conflict_motive', 'N/A')}`")
            st.markdown(f"**Repurposing Target:** `{analysis_data.get('congruence_motive', 'N/A')}`")
            st.json(analysis_data.get("congruence_ratings", {}))
            
        # 1. Get the correct prompt and any auxiliary data (Step 4 preparation)
        prompt_template, injected_data, congruence_ratings = get_prompts_for_condition(
            selected_condition, motive_scores, event_text, analysis_data
        )

        # Display the data injected into the prompt for transparency
        with st.expander("Show Injected Personalization/Appraisal Data (for LLM Guidance)"):
            if injected_data:
                st.code(injected_data, language="text")
            else:
                st.code("No specific personalization or appraisal data was included (Control condition).", language="text")

        # --- STAGE 2: Guidance Generation (Step 4 - LLM Call 2) ---
        guidance = ""
        with st.spinner(f"STAGE 2/2: Generating Guidance using {selected_condition} logic..."):
            
            # LCEL Chain: Prompt | LLM
            chain = prompt_template | llm
            
            try:
                # Execute the chain, using the prompt_template and the single input key 'event_text'
                response = chain.invoke({"event_text": event_text})
                guidance = response.content
                
                # Store data in session state for later submission
                st.session_state.final_guidance = guidance
                st.session_state.analysis_data = analysis_data
                st.session_state.motive_scores = motive_scores
                st.session_state.selected_condition = selected_condition
                st.session_state.event_text = event_text
                
                # 3. Display Result
                st.markdown("### 💬 Repurposing Guidance")
                st.success(guidance)
                st.session_state.show_ratings = True
                
            except Exception as e:
                st.error(f"An error occurred during LLM Guidance generation. Error: {e}")
                st.session_state.show_ratings = False
        
        # 4. Show the Full Prompt Sent to the Model (for research transparency)
        if st.session_state.get('show_ratings', False):
            with st.expander("Show Full System Prompt Sent to Gemini for Guidance (Step 4)"):
                final_prompt = prompt_template.format(event_text=event_text)
                st.code(final_prompt, language="markdown")


# --- Step 5: Participant Rating Collection and Data Submission (Step 6) ---
if st.session_state.get('show_ratings', False) and 'final_guidance' in st.session_state:
    
    st.markdown("---")
    st.markdown("### Step 5 & 6: Collect Participant Ratings and Save")
    st.write("Simulate the participant rating the generated guidance:")
    
    # Initialize ratings in session state if not present
    if "collected_ratings" not in st.session_state:
        st.session_state.collected_ratings = {dim: 3 for dim in RATING_DIMENSIONS}
    
    # Display rating sliders for each dimension
    with st.form("rating_form"):
        
        # Rating Sliders
        for dim in RATING_DIMENSIONS:
            min_val = 1
            max_val = 5
            default_val = 3
            
            # Use specific default for Emotional Valence (mid-point)
            if "Valence" in dim:
                default_val = 3
                
            st.session_state.collected_ratings[dim] = st.slider(f"{dim} (1-5)", min_val, max_val, default_val, key=dim)
        
        # Submit button
        if st.form_submit_button("Submit Ratings and Save Trial Data"):
            
            # Prepare data for Firestore (Step 6)
            trial_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "condition": st.session_state.selected_condition,
                "event_description": st.session_state.event_text,
                "motive_importance_scores": st.session_state.motive_scores,
                "appraisal_analysis": st.session_state.analysis_data,
                "llm_guidance": st.session_state.final_guidance,
                "participant_ratings": st.session_state.collected_ratings,
            }
            
            if save_data(trial_data):
                # Clear state to prepare for the next trial
                del st.session_state.show_ratings
                del st.session_state.final_guidance
                del st.session_state.analysis_data
                del st.session_state.motive_scores
                del st.session_state.collected_ratings
                st.rerun() # Rerun to clear the interface and show success message
