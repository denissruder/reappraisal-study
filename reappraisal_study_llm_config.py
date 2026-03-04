import toml
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict

# Load the configuration
config = toml.load("reappraisal_study_config.toml")

# --- DATA MODELS ---

class InterviewTurn(BaseModel):
    reasoning: str = Field(description="Internal Chain-of-Thought and Verification log.")
    coverage_scores: Dict[str, int] = Field(description="Updated coverage scores (0-10) for all dimensions.")
    conversational_response: str = Field(description="The friendly response or follow-up question for the human.")

class Synthesis(BaseModel):
    reasoning: str = Field(description="Log of data extraction and perspective shifting.")
    final_narrative: str = Field(description="The cohesive first-person narrative.")

# --- WORKER FUNCTIONS ---

def run_interviewer_turn(api_key, history, current_scores):
    llm = ChatGoogleGenerativeAI(model=config["study"]["model_name"], google_api_key=api_key)
    structured_interviewer = llm.with_structured_output(InterviewTurn)

    # Calculate dynamic variables
    q_dict = config["chat"]["questions"]
    count = len(q_dict)
    questions_text = "\n".join([f"- {k.capitalize()}: {v}" for k, v in q_dict.items()])
    
    m_score = config["chat"]["interviewer"]["min_score"]
    m_retries = config["chat"]["interviewer"]["max_retries"]

    # Inject variables into prompt segments
    instruction = config["chat"]["interviewer"]["instruction"].format(
        count=count, questions=questions_text, min_score=m_score
    )
    rules = config["chat"]["interviewer"]["rules"].format(
        min_score=m_score, max_retries=m_retries
    )
    cot = config["chat"]["interviewer"]["chain_of_thought"].format(
        min_score=m_score
    )

    # Build the final prompt
    prompt = config["chat"]["prompts"]["interviewer_template"].format(
        role=config["chat"]["interviewer"]["role"],
        instruction=instruction,
        questions=questions_text,
        coverage_scores="\n".join([f"- {k}: {v}/10" for k, v in current_scores.items()]),
        rules=rules,
        chain_of_thought=cot,
        chain_of_verification=config["chat"]["interviewer"]["chain_of_verification"],
        history="\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history]),
        input=history[-1]["answer"] if history else ""
    )
    
    return structured_interviewer.invoke(prompt)

def run_synthesizer(api_key, history):
    llm = ChatGoogleGenerativeAI(model=config["study"]["model_name"], google_api_key=api_key)
    structured_synth = llm.with_structured_output(Synthesis)

    q_dict = config["chat"]["questions"]
    count = len(q_dict)
    questions_text = "\n".join([f"- {k.capitalize()}: {v}" for k, v in q_dict.items()])

    prompt = config["chat"]["prompts"]["synthesizer_template"].format(
        role=config["chat"]["synthesizer"]["role"],
        instruction=config["chat"]["synthesizer"]["instruction"].format(count=count, questions=questions_text),
        questions=questions_text,
        rules=config["chat"]["synthesizer"]["rules"],
        chain_of_thought=config["chat"]["synthesizer"]["chain_of_thought"].format(count=count, questions=questions_text),
        chain_of_verification=config["chat"]["synthesizer"]["chain_of_verification"],
        history="\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history])
    )
    
    return structured_synth.invoke(prompt)

# --- SATURATION LOGIC ---
def check_interview_saturation(current_scores, previous_scores, stall_counter):
    """
    Evaluates interview completion based on coverage score and saturation percentage.
    """
    # 1. Fetch Thresholds
    m_score = config["chat"]["interviewer"]["min_score"]
    sat_p = config["chat"]["interviewer"].get("saturation_percent", 0.75)
    total_q = len(config["chat"]["questions"])
    
    # 2. DEPTH CHECK: Core foundational dimensions (Event & Feeling)
    # These MUST meet the min_score for the interview to ever finish normally.
    is_deep = (current_scores.get("event", 0) >= m_score and 
               current_scores.get("feeling", 0) >= m_score)
    
    # 3. SATURATION CHECK
    # Calculate how many questions need to pass the min_score (e.g., 8 * 0.75 = 6)
    required_count = max(1, int(total_q * sat_p)) 
    current_saturated_count = len([s for s in current_scores.values() if s >= m_score])
    is_saturated = current_saturated_count >= required_count

    # 4. STALL CHECK (Patience)
    # If the total sum of scores across the vector hasn't increased, increment stall.
    if sum(current_scores.values()) <= sum(previous_scores.values()):
        stall_counter += 1
    else:
        stall_counter = 0

    is_stalled = stall_counter >= config["chat"]["interviewer"]["max_retries"]
    
    # 5. FINAL GATE
    # The user can leave if they have provided deep foundations AND 
    # (they have covered enough other topics OR we've hit a dead end).
    should_finish = is_deep and (is_saturated or is_stalled)
    
    return should_finish, stall_counter
