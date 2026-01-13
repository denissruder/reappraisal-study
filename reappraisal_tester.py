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
    gap: 0.25rem !important; /* Minimal vertical space (4px) */
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
    margin-bottom: 10px !important; 
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
</style>
""", unsafe_allow_html=True)

# --- 1. CONFIGURATION & SETUP ---

MODEL_NAME = "gemini-2.5-flash"
TEMP = 0.8 # Increased temperature for more diverse CoTs
RATING_SCALE_MIN = 1
RATING_SCALE_MAX = 9 
MIN_NARRATIVE_LENGTH = 100
N_COTS = 5 # *** Number of Chain-of-Thought runs for Self-Consistency ***

# CORE DATA: Single source of truth for all motives and their goals
MOTIVES_GOALS = [
    ("Hedonic", "To feel good", "Not to feel bad", "The need to experience pleasure and comfort, or to avoid pain and discomfort."),
    ("Physical", "To be in good health", "To stay safe", "The need to maintain one's body (health, fitness) and ensure its continued well-being."),
    ("Wealth", "To have money", "To avoid poverty", "The need for financial resources and economic security."),
    ("Predictability", "To understand", "To avoid confusion", "The need to understand the environment, have order, and avoid uncertainty or confusion."),
    ("Competence", "To succeed", "To avoid failure", "The need to feel capable, effective, and successful at tasks and challenges."),
    ("Growth", "To learn and grow", "To avoid monotony or decline", "The need for continuous learning, development, and personal expansion."),
    ("Autonomy", "To be free to decide", "Not to be told what to do", "The need for self-determination, control over one's life, and the ability to make independent choices."),
    ("Relatedness", "To feel connected", "To avoid loneliness", "The need to form and maintain close, positive, and meaningful bonds with others."),
    ("Acceptance", "To be liked", "To avoid disapproval", "The need to be approved of, respected, and feel included by social groups."),
    ("Status", "To stand out", "To avoid being ignored", "The need for recognition, social rank, influence, or prominence within a group."),
    ("Responsibility", "To live up to expectations", "Not to let others down", "The need to fulfill one's duties, meet obligations, and act reliably towards others."),
    ("Meaning", "To make a difference", "Not to waste my life", "The need to feel that one's life is significant, purposeful, and contributes to something larger."),
    ("Instrumental", "To gain something", "To avoid something", "The need to view actions and resources as means to an end, focused on practical outcomes."),
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
for motive_name, _, _, _ in MOTIVES_GOALS:
    MOTIVE_SCORE_KEYS.append(f"{motive_name}_Promotion")
    MOTIVE_SCORE_KEYS.append(f"{motive_name}_Prevention")

JSON_KEYS_LIST = ", ".join(MOTIVE_SCORE_KEYS)



# --- FOR STREAMLIT PAGE LOGIC ---
MOTIVES_FULL = [
    {'motive': m[0], 'Promotion': m[1], 'Prevention': m[2], 'Definition': m[3]}
    for m in MOTIVES_GOALS
]

# --- Few-Shot Examples ---

# FEW-SHOT EXAMPLE for Motive Prediction
example1 = """
Event: I am falsely accused of a serious, high-profile intellectual property theft by a former business partner. The initial police report and subsequent media coverage have already ruined my professional reputation, causing clients to drop me instantly and my business to implode overnight. I’ve had to mortgage my home to cover the massive legal defense costs, retaining a top-tier lawyer just to have a chance against the well-funded opponent. The case is moving slowly, and the discovery phase requires me to spend countless hours reviewing old documents and communications, reliving the breakdown of the partnership. I know I am innocent, but the burden of proof, the aggressive legal tactics of the other side, and the sheer length of the process are mentally debilitating. I receive constant, hostile messages online from strangers who have read the biased media reports. My savings are gone, my career is a wreck, and the stress has led to profound marital strain. The anticipation of the trial and the potential for a devastating verdict, despite the truth being on my side, makes every morning a struggle to simply get out of bed.

Hedonic_Promotion : (1, The individual is currently in a state of crisis management; seeking pleasure or 'feeling good' is not a priority compared to survival),
Hedonic_Prevention : (9, The primary emotional state is avoiding profound mental distress, debilitating stress, and the misery of the current situation.),

Physical_Promotion : (2, There is little focus on physical optimization or health improvement.),
Physical_Prevention : (7, The stress is described as 'mentally debilitating,' suggesting a high priority on preventing a complete physical or nervous breakdown.),

Wealth_Promotion : (2, The person is not trying to get rich; they are in a defensive financial posture.),
Wealth_Prevention : (9, The loss of savings, the mortgaged home, and the 'business imploding' make avoiding total poverty a top-tier motive.),

Predictability_Promotion : (3, There is less focus on gaining new understanding and more on the uncertainty of the legal outcome.),
Predictability_Prevention : (8, The 'anticipation of the trial' and the slow process create a desperate need to avoid the confusion and chaos of the legal system.),

Competence_Promotion : (6, The desire to eventually succeed and be exonerated is present but overshadowed by defensive needs.),
Competence_Prevention : (9, Avoiding 'failure' in the form of a 'devastating verdict' is a matter of professional and personal survival.),

Growth_Promotion : (1, The situation is about survival, not personal growth or learning; the person is reliving a breakdown, not moving forward.),
Growth_Prevention : (9, The primary focus is stopping the 'wreck' of a career and avoiding the further decline of their life's work.),

Autonomy_Promotion : (4, The person desires their life back, but they are currently trapped by the legal process.),
Autonomy_Prevention : (9, They are being forced to spend 'countless hours' on discovery and are reacting to 'aggressive legal tactics,' creating a strong desire not to be controlled by the opponent.),

Relatedness_Promotion : (3, While they likely want connection, the 'marital strain' suggests they are currently unable to focus on building intimacy.),
Relatedness_Prevention : (9, A very high focus on avoiding the 'loneliness' caused by the withdrawal of clients and the strain on the marriage.),

Acceptance_Promotion : (4, Gaining new friends is irrelevant; they want their old life back.),
Acceptance_Prevention : (9, The 'ruined professional reputation' and 'hostile messages from strangers' make avoiding social disapproval and being an outcast a dominant motive.),

Status_Promotion : (3, The person is not trying to 'stand out' anymore; they are trying to recover what was lost.),
Status_Prevention : (9, Avoiding the shame of being labeled a 'thief' and the public stigma of the 'biased media reports' is critical.),

Responsibility_Promotion : (5, The person wants to live up to the image of an innocent partner/spouse.),
Responsibility_Prevention : (9, A major driver is not 'letting down' their family, especially given the financial risks like the mortgaged home.),

Meaning_Promotion : (7, The person is motivated by the 'truth being on my side,' suggesting a desire to ensure that truth and justice prevail.),
Meaning_Prevention : (8, The struggle to 'get out of bed' suggests a fear that their life and hard work are being 'wasted' by a lie.),

Instrumental_Promotion : (8, The person is working toward a specific goal: a top-tier lawyer and a trial victory.),
Instrumental_Prevention : (9, The entire current existence is defined by the need to avoid a 'devastating verdict' and the total loss of their assets.)

"""
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

def get_majority_vote(scores_list):
    """Returns the most frequent score. In case of a tie, returns the highest."""
    if not scores_list:
        return 1
    counts = Counter(scores_list)
    max_freq = max(counts.values())
    # Find all scores that appeared with the maximum frequency
    modes = [score for score, count in counts.items() if count == max_freq]
    # Return the highest score among the modes (conservative tie-breaking)
    return max(modes)

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
            placeholder = "I was supposed to give a major presentation to my client, and just minutes before, my laptop crashed, losing several hours of preparation. I had to improvise everything on a backup system. I felt incompetent, and worried I would lose the client's business, which would reflect poorly on my whole team."
            return placeholder
            
    except Exception as e:
        st.warning(f"Failed to fetch random story from database: {e}. Using a fallback placeholder.")
        return "Database fetch failed. Placeholder: I was supposed to give a major presentation to my client, and just minutes before, my laptop crashed, losing several hours of preparation. I had to improvise everything on a backup system. I felt incompetent, and worried I would lose the client's business, which would reflect poorly on my whole team."

# --- 2. LLM LOGIC FUNCTIONS (Self-Consistency/Multiple CoTs) ---

# --- LLM APPRAISAL PREDICTION TEMPLATE ---

MOTIVES_PREDICTION_TEMPLATE = f"""

You are an expert in human psychology. 

--- Chain-of-Thought ---

1. You will read a related theory, list of motives with their promotion and prevention focus pairs, and a scale to be assigned for each motive.

2. You will read a first-hand description of an event that elicited an emotional reaction.

3. You will read a few output examples that you need to follow.

4. Your will guess how important motives were to a person may have considered in a given event.  


-- PHASE 1: Chain-of-Thought & Chain-of-Verification Reasoning --- 

A motive is an outcome that the person desires to approach or to avoid.  

People usually have many motives, they desire to approach or avoid many things in life. However, within a given situation, usually only a subset of all possible motives are important to the person.  

The person may or may not be consciously aware of this desire. Conscious awareness is not required for a motive to exert force on behavior, that is to explain why someone behaved the way they did. This means that some of the momentarily important motives can be verbally expressed in the texts you read. However, other may remain somewhat hidden and you can try to infer their importance from the context and the indirect clues the text provides.  

There are 13 core motives organized in closely related promotion and prevention pairs.  

List of core motives: Hedonic, Physical, Wealth, Predictability, Competence, Growth, Autonomy, Relatedness, Acceptance, Status, Responsibility, Meaning, Instrumental

List of promotion motives: 
    Hedonic_Promotion : To feel good, 
    Physical_Promotion : To be in good health,
    Wealth_Promotion : To have money,
    Predictability_Promotion : To understand,
    Competence_Promotion : To succeed,
    Growth_Promotion : To learn and grow,
    Autonomy_Promotion : To be free to decide,
    Relatedness_Promotion : To feel connected,
    Acceptance_Promotion : To be liked,
    Status_Promotion : To stand out,
    Responsibility_Promotion : To live up to expectations,
    Meaning_Promotion : To make a difference,
    Instrumental_Promotion : To gain something

List of prevention motives: 
    Hedonic_Prevention: To avoid feeling bad,
    Physical_Prevention: To avoid injury or illness,
    Wealth_Prevention: To avoid losing money,
    Predictability_Prevention: To avoid confusion or chaos,
    Competence_Prevention: To avoid failure,
    Growth_Prevention: To avoid stagnation or regressing,
    Autonomy_Prevention: To avoid being controlled by others,
    Relatedness_Prevention: To avoid feeling lonely or isolated,
    Acceptance_Prevention: To avoid being rejected or disliked,
    Status_Prevention: To avoid losing face or reputation,
    Responsibility_Prevention: To avoid letting others down,
    Meaning_Prevention: To avoid leading a pointless life

Motives are rated on a scale from {RATING_SCALE_MIN} (Not at all important) to **{RATING_SCALE_MAX} (Very important) to determine their dominant focus.

-- PHASE 2: Chain-of-Thought & Chain-of-Verification Reasoning --- 

Event: {{event_text}} 

--- PHASE 3: Chain-of-Thought & Chain-of-Verification Reasoning --- 

Example 1: {{example1}}

--- PHASE 4: Chain-of-Thought & Chain-of-Verification Reasoning --- 

Given a theory and motive list from PHASE 1, event description from PHASE 2, and output examples from PHASE 3, your main task is to guess how important each motive was to a person in this event.   

--- Chain-of-Verification ---

1. Make sure that you clearly understand the theory, motive dimensions, and that all ratings are on a scale from {RATING_SCALE_MAX} (Not Relevant At All) to {RATING_SCALE_MAX} (Highly Relevant) scale.

2. Make sure that the event text is meaningful and wasn't copied from the internet. 

3. Make sure you follow the exact tuple format from the examples: Motive_Type : (Score, Concise justification).

4. Make sure that your reasoning for each motive dimension is objective and ethnically correct.

"""
        
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
            st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
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

    with st.form("initial_assessment_form"):
        st.markdown("### General Motive Importance & Focuses")
        st.markdown(f"**1 = Not Important At All** | **{RATING_SCALE_MAX} = Extremely Important**")
        st.markdown("<br>", unsafe_allow_html=True)

        motive_scores = st.session_state.general_motive_scores
        for m in MOTIVES_FULL:
            # Removed Title, Definition, and HR to collapse the row
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

        if st.form_submit_button("Next: Start Interview", type="primary"):
            st.session_state.page = 'chat' 
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
    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1)) 

    if 'situation_motive_scores' not in st.session_state:
        st.session_state.situation_motive_scores = {
            m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL
        }

    with st.form("situation_rating_form"):
        st.markdown("### Situation Appraisal: Your Perspectives")
        st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #FFF;'>", unsafe_allow_html=True)
        st.markdown(f"""
        Please rate how **relevant** each of the following motives and their focuses was to the **event you just described** on a scale of 1 to {RATING_SCALE_MAX} 
        """)
        st.markdown(f"**1 = Not Important At All** | **{RATING_SCALE_MAX} = Extremely Important**")

        motive_scores = st.session_state.situation_motive_scores
        for m in MOTIVES_FULL:
            st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)
            # REMOVED: {m['Definition']}
            st.markdown(f"#### {m['motive']}")
            
            col1, col2 = st.columns(2) 
            
            with col1:
                motive_scores[m['motive']]['Promotion'] = st.radio(
                    f"{m['Promotion']}", 
                    options=RADIO_OPTIONS, 
                    index=motive_scores[m['motive']]['Promotion'] - 1, 
                    horizontal=True, 
                    key=f"sit_{m['motive']}_Promotion" # Changed key to be unique from 'gen_'
                )
            
            with col2:
                motive_scores[m['motive']]['Prevention'] = st.radio(
                    f"{m['Prevention']}", 
                    options=RADIO_OPTIONS, 
                    index=motive_scores[m['motive']]['Prevention'] - 1, 
                    horizontal=True, 
                    key=f"sit_{m['motive']}_Prevention"
                )

        if st.form_submit_button("Next: Cross-Participant Rating", type="primary"):
            st.session_state.page = 'cross_rating'
            st.rerun()

def show_cross_rating_page():
    # --- 1. Story Persistence ---
    if 'cross_participant_situation' not in st.session_state:
        try:
            st.session_state.cross_participant_situation = get_random_story_from_db()
        except:
            st.session_state.cross_participant_situation = "Sample story context..."
    
    if 'cross_submitted' not in st.session_state:
        st.session_state.cross_submitted = False

    # --- BRANCH A: PROCESSING STATE (Stage Counter 1/5, 2/5...) ---
    if st.session_state.cross_submitted:
        # JS Auto-scroll to top
        st.markdown("<script>window.parent.document.querySelector('.main').scrollTo(0,0);</script>", unsafe_allow_html=True)
        
        st.header("🧬 Expert Consensus Engine")
        
        # UI Feedback elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.status("Analyzing narrative...", expanded=True) as status:
            # FIX: Define the chain here to ensure it's in scope
            prompt = PromptTemplate(
                input_variables=["event_text", "example1"], 
                template=MOTIVES_PREDICTION_TEMPLATE
            )
            # Ensure 'llm' is defined globally (e.g., llm = ChatGoogleGenerativeAI(...))
            chain = prompt | llm 
            
            N_COTS = 5
            all_valid_runs = []
            
            for i in range(N_COTS):
                step = i + 1
                status_text.markdown(f"**Stage {step}/{N_COTS}:** Consulting Reasoning Path {step}...")
                progress_bar.progress(step / N_COTS)
                
                try:
                    # Invoke the chain
                    response = chain.invoke({
                        "event_text": st.session_state.final_event_narrative,
                        "example1": example1 # Ensure 'example1' is a global string
                    })
                    
                    # Parse using your Tuple Regex Parser
                    scores, reasonings = parse_llm_json(response.content, step)

                    if scores:
                        all_valid_runs.append({"scores": scores, "reasoning": reasonings})
                        st.write(f"✅ Path {step}/{N_COTS} completed.")
                    else:
                        st.write(f"⚠️ Path {step}/{N_COTS} failed parsing.")
                
                except Exception as e:
                    st.write(f"❌ Path {step}/{N_COTS} error: {e}")
                
                time.sleep(0.2)

            if all_valid_runs:
                status_text.markdown("**Finalizing:** Computing Majority Vote winner...")
                
                # Calculate Majority Vote Consensus
                consensus_scores = {}
                for key in MOTIVE_SCORE_KEYS:
                    votes = [run["scores"][key] for run in all_valid_runs if key in run["scores"]]
                    consensus_scores[key] = get_majority_vote(votes)

                # Assemble Final Flattened Data
                trial_data = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "participant_id": str(uuid.uuid4()),
                    
                    # Flattened maps (Motive_Focus: Score)
                    "baseline_motive_profile": flatten_motive_dict(st.session_state.general_motive_scores),
                    "baseline_regulatory_focus": flatten_motive_dict(st.session_state.reg_focus_scores),
                    "own_situation_rating": flatten_motive_dict(st.session_state.situation_motive_scores),
                    "cross_perspective_rating": flatten_motive_dict(st.session_state.cross_motive_scores),
                    
                    # LLM Data
                    "llm_consensus_prediction": consensus_scores,
                    "llm_all_reasoning_paths": all_valid_runs, # All 5 paths (scores + reasoning maps)
                    
                    "confirmed_event_narrative": st.session_state.final_event_narrative,
                    "cross_situation_text": st.session_state.cross_participant_situation
                }
                
                if save_data(trial_data):
                    status.update(label="Analysis Complete! Redirecting...", state="complete")
                    st.session_state.page = 'thank_you'
                    st.rerun()
            else:
                st.error("Consensus could not be reached. All paths failed.")
                if st.button("Retry"):
                    st.session_state.cross_submitted = False
                    st.rerun()
        return

    # --- BRANCH B: INPUT STATE (The Form) ---
    st.header("🎯 Perspective Taking")
    st.info(st.session_state.cross_participant_situation)

    RADIO_OPTIONS = list(range(1, RATING_SCALE_MAX + 1))
    if 'cross_motive_scores' not in st.session_state:
        st.session_state.cross_motive_scores = {m['motive']: {'Promotion': 5, 'Prevention': 5} for m in MOTIVES_FULL}

    with st.form("cross_rating_form"):
        for m in MOTIVES_FULL:
            # Cleaned up the label for consistency
            st.markdown(f"#### {m['motive']}")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.cross_motive_scores[m['motive']]['Promotion'] = st.radio(
                    f"{m['Promotion']}", options=RADIO_OPTIONS, horizontal=True,
                    index=st.session_state.cross_motive_scores[m['motive']]['Promotion'] - 1,
                    key=f"c_{m['motive']}_pro"
                )
            with col2:
                st.session_state.cross_motive_scores[m['motive']]['Prevention'] = st.radio(
                    f"{m['Prevention']}", options=RADIO_OPTIONS, horizontal=True,
                    index=st.session_state.cross_motive_scores[m['motive']]['Prevention'] - 1,
                    key=f"c_{m['motive']}_pre"
                )
            st.markdown("<hr style='margin: 5px 0 15px 0; border: 0.5px solid #eee;'>", unsafe_allow_html=True)

        if st.form_submit_button("Submit All Data and Finish Trial", type="primary"):
            st.session_state.cross_submitted = True
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
elif st.session_state.page == 'cross_rating':
    show_cross_rating_page()
elif st.session_state.page == 'thank_you': 
    show_thank_you_page()
