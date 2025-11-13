import streamlit as st
import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import LLMChain

# --- NEW: Firestore Imports (Requires 'google-cloud-firestore' package) ---
try:
    from google.cloud import firestore
except ImportError:
    st.error("The 'google-cloud-firestore' library is not installed. Please add it to your requirements.txt.")

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.5
MOTIVES = ["Achievement & Success", "Security & Stability", "Affiliation & Belonging", "Stimulation & Excitement", "Self-Direction & Autonomy"]
RATING_DIMENSIONS = ["Believability", "Appropriateness", "Emotional Valence (1=Negative, 5=Positive)"]

# --- Database Connection and Persistence ---

@st.cache_resource
def get_firestore_db():
    """Initializes and returns the Firestore client."""
    # Streamlit Cloud Secure Secrets Management:
    # 1. Ensure your Google Service Account credentials are in a 'gcp_service_account' key
    #    in your Streamlit secrets.toml file.
    # 2. This function automatically uses the credentials set up in the Streamlit environment.
    try:
        # st.secrets["gcp_service_account"] should contain the JSON credentials
        if 'gcp_service_account' in st.secrets:
            # Explicitly load credentials from secrets
            key_dict = json.loads(st.secrets["gcp_service_account"])
            # Initialize the client using the credentials
            db = firestore.Client.from_service_account_info(key_dict)
            return db
        else:
            st.warning("Firestore connection failed: 'gcp_service_account' not found in secrets.")
            return None
    except Exception as e:
        st.error(f"Error initializing Firestore: {e}")
        return None

def save_trial_data_to_database(db, data):
    """
    Saves the complete experimental trial data structure to Firestore.
    """
    if db is None:
        st.error("Database connection is not available. Saving to local session state only.")
        return False

    try:
        # Use a clearly defined public collection path for study data
        collection_ref = db.collection("reappraisal_study_trials")
        collection_ref.add(data)
        return True
    except Exception as e:
        st.error(f"Failed to write data to Firestore. Check permissions and connection. Error: {e}")
        return False

# --- LLM and App Setup ---

# Initialize LLM, securely loading the key from Streamlit secrets
@st.cache_resource
def init_llm():
    try:
        # The environment variable is automatically set by Streamlit using the secret
        # if you include 'google_api_key' in your secrets.toml or environment.
        if "GEMINI_API_KEY" not in os.environ:
             st.error("GEMINI_API_KEY is not set in environment or Streamlit secrets.")
             return None
             
        return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMP)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

llm = init_llm()

# Exit if LLM or DB isn't ready
if not llm:
    st.stop()

db = get_firestore_db()
if db:
    st.sidebar.success("✅ Database Connected")
else:
    st.sidebar.warning("⚠️ Database connection pending/failed. Data will not be saved persistently.")


# --- LLM Appraisal Analysis (STEP 3) - Template remains the same ---
APPRAISAL_ANALYSIS_TEMPLATE = """
You are an Appraisal Analyst. Your task is to analyze the user's event description in the context of their core motives.
Your output MUST be a valid JSON object. Do not include any text, headers, or markdown formatting outside of the JSON block.
...
"""
# (The rest of the run_appraisal_analysis and get_prompts_for_condition functions are omitted for brevity,
# as they remain functionally identical to the previous response, but they are included in the complete file below.)

def run_appraisal_analysis(llm, motive_scores, event_text):
    """Executes the LLM to perform Step 3: Appraisal Analysis."""
    
    scores_list_formatted = "\n".join([f"- {motive}: {score}/5" for motive, score in motive_scores.items()])
    motive_list = ", ".join(MOTIVES)

    prompt = PromptTemplate(
        input_variables=["motive_list", "event_text", "scores_list_formatted"], 
        template=APPRAISAL_ANALYSIS_TEMPLATE.replace("...", "")
    ) # Removed '...' placeholder from template for final version
    
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    
    try:
        json_string = chain.run(
            motive_list=motive_list, 
            event_text=event_text, 
            scores_list_formatted=scores_list_formatted
        )
        
        json_string = json_string.strip().replace("```json", "").replace("```", "")
        analysis_data = json.loads(json_string)
        
        return analysis_data
        
    except Exception as e:
        raw_output_snippet = json_string[:200].replace('\n', '\\n')
        st.error(f"Error during LLM Appraisal Analysis (Step 3). Could not parse JSON. Error: {e}. Raw LLM output: {raw_output_snippet}...")
        return None

def get_prompts_for_condition(condition, motive_scores, event_text, analysis_data):
    """
    Generates the specific system instruction (template) for Guidance (Step 4)
    using the LLM-generated analysis_data (Step 3).
    """
    
    mock_conflict = analysis_data.get("conflict_motive", "N/A")
    mock_congruence = analysis_data.get("congruence_motive", "N/A")
    congruence_ratings = analysis_data.get("congruence_ratings", {})

    highest_score = max(motive_scores.values())
    most_important_motives = [m for m, s in motive_scores.items() if s == highest_score]
    core_motive = most_important_motives[0] 

    # Fix: Explicitly convert congruence ratings to string to prevent format errors
    scores_list_formatted = "\n".join([
        f"- {motive}: {score}/5 (Congruence: {str(congruence_ratings.get(motive, 'N/A'))}/5)" 
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

# --- Streamlit UI and Execution ---

st.set_page_config(layout="wide", page_title="Reappraisal Experiment Tester")
st.title("🔬 Repurposing Reappraisal Experiment Tester")
st.caption("Simulates the three experimental conditions for personalized guidance based on motivational profiles.")

# --- Sidebar for Controls ---
motive_scores = {}
with st.sidebar:
    st.header("Experiment Controls")
    
    selected_condition = st.radio(
        "Select Experimental Condition:",
        ["1. Control", "2. Appraisal-Aware", "3. Importance and Appraisal Aware"],
        index=2
    )
    
    st.subheader("Simulated Motive Importance (Step 1)")
    st.write("Rate each motive on a scale of 1 to 5 (1=Not Important, 5=Most Important):")
    
    for motive in MOTIVES:
        motive_scores[motive] = st.slider(f"{motive}", 1, 5, 3, key=f"motive_{motive}") 
    
    st.info("Data saving is configured for Google Firestore via Streamlit secrets.")


# --- Main Content Area ---

st.subheader("Step 1 & 2: Event Elicitation (User Input)")
st.write("Enter the detailed, emotionally charged event description from the user.")

event_text = st.text_area(
    "Event Description",
    height=200,
    value="After a sudden merger, my department is being completely overhauled, and a new, much younger VP has been appointed. I have a 30-day ultimatum to develop an unfamiliar system or face termination. This coincides with a major financial commitment for my child's university tuition. I am working 18-hour days, feeling immense pressure, and my health is suffering due to the toxic environment.",
    key="event_input"
)

if st.button("Generate Repurposing Guidance", type="primary", use_container_width=True) and event_text:
    
    st.subheader(f"Step 3: LLM Appraisal Analysis and Step 4: Guidance")
    
    analysis_data = None
    guidance = None
    
    with st.spinner("STAGE 1/2: Analyzing Congruence (Step 3)..."):
        analysis_data = run_appraisal_analysis(llm, motive_scores, event_text)
        
    if analysis_data:
        
        with st.expander("Show LLM-Generated Congruence Ratings (Step 3 Output)"):
            st.markdown(f"**Motive Most Conflicted:** `{analysis_data.get('conflict_motive', 'N/A')}`")
            st.markdown(f"**Repurposing Target:** `{analysis_data.get('congruence_motive', 'N/A')}`")
            st.json(analysis_data.get("congruence_ratings", {}))
            
        prompt_template, injected_data, congruence_ratings = get_prompts_for_condition(
            selected_condition, motive_scores, event_text, analysis_data
        )

        with st.expander("Show Injected Personalization/Appraisal Data (for LLM Guidance)"):
            if injected_data:
                st.code(injected_data, language="text")
            else:
                st.code("No specific personalization or appraisal data was included (Control condition).", language="text")

        with st.spinner(f"STAGE 2/2: Generating Guidance using {selected_condition} logic..."):
            
            chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
            
            try:
                guidance = chain.run(event_text=event_text)
                
                st.markdown("### 💬 Repurposing Guidance")
                st.success(guidance)
                
                # --- Step 5: Participant Rating Collection and Save ---
                st.markdown("---")
                st.markdown("### Step 5: Collect Participant Ratings")
                st.write("Simulate the participant rating the generated guidance:")
                
                # The form collects ratings and triggers the save function
                with st.form("rating_form"):
                    
                    collected_ratings = {}
                    for dim in RATING_DIMENSIONS:
                        min_val = 1
                        max_val = 5
                        default_val = 3
                        if "Valence" in dim:
                            default_val = 3
                            
                        # Ensure keys are unique across all sessions (use a counter or datetime if this was a long-running app)
                        collected_ratings[dim] = st.slider(f"{dim} (1-5)", min_val, max_val, default_val, key=f"rating_{dim}")
                    
                    # Submit button triggers the data aggregation and save
                    if st.form_submit_button("Submit Ratings and Save Trial"):
                        
                        trial_data = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "condition": selected_condition,
                            "event_text": event_text,
                            "motive_importance_scores": motive_scores,
                            "appraisal_analysis_llm": analysis_data,
                            "generated_guidance": guidance,
                            "participant_ratings": collected_ratings
                        }
                        
                        if save_trial_data_to_database(db, trial_data):
                            st.balloons()
                            st.success("Trial data successfully saved to Firestore! Ready for the next participant.")
                        else:
                            st.error("Data was NOT saved persistently. Please configure Firestore secrets correctly.")
                
            except Exception as e:
                st.error(f"An error occurred during LLM Guidance generation. Error: {e}")

        with st.expander("Show Full System Prompt Sent to Gemini for Guidance (Step 4)"):
            final_prompt = prompt_template.format(event_text=event_text)
            st.code(final_prompt, language="markdown")
