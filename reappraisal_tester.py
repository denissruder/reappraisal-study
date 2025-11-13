import streamlit as st
import os
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# FIX: LLMChain is deprecated. Using LangChain Expression Language (LCEL) instead.
# The necessary components (PromptTemplate, ChatGoogleGenerativeAI) are already imported.

# --- NEW: Firestore Imports (Requires 'google-cloud-firestore' package) ---
try:
    from google.cloud import firestore
except ImportError:
    st.error("The 'google-cloud-firestore' package is not installed. Please check requirements.txt.")
    st.stop()

# --- 1. CONFIGURATION ---

# 1.1 Securely load Firebase credentials and initialize Firestore client
# Reads the 'gcp_service_account' secret (JSON dictionary)
@st.cache_resource
def get_firestore_client():
    if "gcp_service_account" not in st.secrets:
        st.error("Database Error: 'gcp_service_account' secret not found. Check Streamlit Secrets configuration.")
        st.stop()
        
    try:
        # Load the dictionary from the multi-line string secret
        key_dict = json.loads(st.secrets["gcp_service_account"])
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
    
    # Initialize the LLM
    try:
        # Use a low temperature for consistent, factual outputs suitable for a study
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        st.sidebar.success("✅ LLM Initialized")
        return llm
    except Exception as e:
        st.sidebar.error(f"❌ LLM Initialization Failed: {e}")
        st.stop()

llm = get_llm()


# --- 2. PROMPTS (LCEL Components) ---

# 2.1 Prompt for NEUTRAL Reappraisal
NEUTRAL_TEMPLATE = """
You are a **Neutral Reappraisal Assistant** designed for research. Your task is to process the user's detailed description of an event and provide widely accepted, low-effort stress management advice using ONLY objective, factual, and emotionally neutral language.

Your output must function as a neutral reframing of the user's situation.

Guidelines for Neutral Reappraisal:
1. Extract only the verifiable facts, conditions, and actions.
2. Eliminate all emotional language.
3. Replace qualitative descriptions with quantitative facts where possible.
4. The final output must be a single, objective summary of the situation's operational facts.
5. Tone: Objective, general, and slightly detached.
6. Content: A simple suggestion focused on action, perspective-taking, or seeking support.

User's Event Description: {event_description}
Neutral Reappraisal:
"""
NEUTRAL_PROMPT = PromptTemplate(input_variables=["event_description"], template=NEUTRAL_TEMPLATE)

# 2.2 Prompt for POSITIVE Reappraisal
POSITIVE_TEMPLATE = """
You are a **Positive Reappraisal Coach** designed for research. Your task is to process the user's detailed description of an event and provide motivational, optimistic guidance that reframes the situation as a challenge or opportunity for growth.

Your output must function as a positive reframing of the user's situation.

Guidelines for Positive Reappraisal:
1. Reframe facts as opportunities for learning or resilience.
2. Emphasize the user's ability to cope and succeed.
3. Use encouraging, high-energy, and hopeful language.
4. The final output must be an inspiring summary of the situation's potential for a positive outcome.
5. Tone: Motivational, enthusiastic, and supportive.
6. Content: A suggestion focused on personal strength, future success, or leveraging resources.

User's Event Description: {event_description}
Positive Reappraisal:
"""
POSITIVE_PROMPT = PromptTemplate(input_variables=["event_description"], template=POSITIVE_TEMPLATE)


# --- 3. CORE PROCESSING CHAIN (LCEL) ---

# Creates the LCEL chain (Prompt | LLM) that handles the processing.
# We create a function to select the appropriate chain based on user choice.
def create_chain(condition):
    if condition == "Neutral Reappraisal":
        return NEUTRAL_PROMPT | llm
    elif condition == "Positive Reappraisal":
        return POSITIVE_PROMPT | llm
    return None

# Caches the LLM generation to prevent re-running on small app interactions
@st.cache_data
def generate_reappraisal(event_description, condition):
    chain = create_chain(condition)
    if not chain:
        return "Error: Please select a reappraisal condition."
    
    # Use .invoke() for a single input/output call (standard LCEL execution)
    response = chain.invoke({"event_description": event_description})
    
    # The response object is a LangChain message; extract the content string
    return response.content


# --- 4. DATA SAVING LOGIC ---

def save_data(data):
    """Saves the data dictionary as a new document in Firestore."""
    try:
        # Use add() to automatically generate a unique document ID
        db.collection(COLLECTION_NAME).add(data)
        st.success("✅ Trial data saved successfully to Firestore!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save data: {e}. Check Firestore rules and credentials.")
        return False


# --- 5. STREAMLIT INTERFACE ---

st.title("🧠 Reappraisal Strategy Study")
st.markdown("---")

# 5.1 Sidebar for Experiment Configuration
st.sidebar.header("Experiment Setup")

# Set up the condition selection
condition = st.sidebar.selectbox(
    "1. Select Reappraisal Condition:",
    ("Neutral Reappraisal", "Positive Reappraisal")
)
st.sidebar.markdown("---")


# 5.2 Main Content Area: Event Input
st.subheader(f"Condition: {condition}")

event_description = st.text_area(
    "Describe a recent, challenging, or stressful event in detail:",
    key="event_input",
    height=200,
    placeholder="Example: I have been working 18-hour days to meet a client deadline, and I worry about the quality of my output and missing my child's recital."
)

if st.button("Generate Reappraisal Guidance", type="primary") and event_description:
    st.session_state.reappraisal_text = generate_reappraisal(event_description, condition)
    st.session_state.event_description = event_description
    st.session_state.condition = condition
    st.session_state.show_ratings = True
else:
    # Ensure ratings are hidden if no text is generated or button is pressed without text
    if "show_ratings" not in st.session_state:
        st.session_state.show_ratings = False
    

# 5.3 Display Guidance and Ratings
if st.session_state.get('show_ratings', False) and 'reappraisal_text' in st.session_state:
    
    st.markdown("### LLM Guidance")
    st.info(st.session_state.reappraisal_text)
    
    st.markdown("### Participant Ratings (1-5 Scale)")
    
    # Define rating dimensions
    dimensions = {
        "helpfulness": "How helpful was the guidance in reframing the event?",
        "emotional_impact": "How much did the guidance reduce your negative emotional response?",
        "feasibility": "How feasible is the suggested action or perspective change?",
        "personal_relevance": "How personally relevant did the guidance feel to your situation?"
    }
    
    # Use session state to hold rating values
    if "ratings" not in st.session_state:
        st.session_state.ratings = {key: 3 for key in dimensions.keys()}

    col1, col2 = st.columns(2)
    
    # Display sliders for ratings
    for i, (key, question) in enumerate(dimensions.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            # Use a key based on the dimension key
            st.session_state.ratings[key] = st.slider(
                label=question,
                min_value=1,
                max_value=5,
                value=st.session_state.ratings[key],
                step=1,
                key=f"rating_{key}"
            )

    # Submission button
    if st.button("Submit Ratings and Save Trial", type="secondary"):
        
        # Prepare data for Firestore
        trial_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "condition": st.session_state.condition,
            "event_description": st.session_state.event_description,
            "llm_guidance": st.session_state.reappraisal_text,
            "participant_ratings": st.session_state.ratings,
        }
        
        if save_data(trial_data):
            # Clear state to prepare for the next trial
            st.session_state.show_ratings = False
            st.session_state.reappraisal_text = None
            st.session_state.event_description = ""
            st.session_state.ratings = {key: 3 for key in dimensions.keys()} # Reset ratings
            st.rerun() # Rerun to clear the interface and show success message
