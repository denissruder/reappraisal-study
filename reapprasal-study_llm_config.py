import toml
from pydantic import BaseModel, Field
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the Brain
config = toml.load("reapprasal-study_config.toml")

# --- SCHEMAS ---
class InterviewTurn(BaseModel):
    reasoning: str = Field(description="Internal CoT and CoVe for interviewer.")
    coverage_scores: Dict[str, int] = Field(description="Scores 0-10 for the 8 core questions.")
    conversational_response: str = Field(description="The response shown to the user.")
    status: str = Field(description="'continue' or 'complete' based on saturation.")

class SynthesisResponse(BaseModel):
    reasoning: str = Field(description="Internal CoT and CoV for synthesizer.")
    final_narrative: str = Field(description="The full first-person story summary.")

# --- HELPERS ---
def get_questions_text():
    return "\n".join([f"- {v}" for v in config["chat"]["questions"].values()])

# --- WORKERS ---
def run_interviewer_turn(api_key, history, current_scores):
    llm = ChatGoogleGenerativeAI(model=config["study"]["model_name"], google_api_key=api_key)
    structured_interviewer = llm.with_structured_output(InterviewTurn)

    # Prepare the standard question list
    questions_list = get_questions_text()

    # NESTED INJECTION: Populate the Instruction
    instruction = config["chat"]["interviewer"]["instruction"].format(questions=questions_list)

    # Prepare scores list
    scores_readable = "\n".join([f"- {k}: {v}/10" for k, v in current_scores.items()])
    
    # ASSEMBLE PROMPT
    prompt = config["chat"]["prompts"]["interviewer_template"].format(
        role = config["chat"]["interviewer"]["role"],
        instruction = instruction,
        questions = questions_list,         
        coverage_scores = scores_readable,
        rules = config["chat"]["interviewer"]["rules"],
        chain_of_thought = config["chat"]["interviewer"]["chain_of_thought"],
        chain_of_verification = config["chat"]["interviewer"]["chain_of_verification"],
        history = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history]),
        input = history[-1]["answer"] if history else ""
    )
    
    return structured_interviewer.invoke(prompt)

def run_synthesizer(api_key, history):
    # Lower temperature for faithful extraction
    llm = ChatGoogleGenerativeAI(model=config["study"]["model_name"], google_api_key=api_key, temperature=0.3)
    structured_synthesizer = llm.with_structured_output(SynthesisResponse)

    # Prepare the standard question list
    questions_list = get_questions_text()
    
    # NESTED INJECTION: Populate the Instruction
    instruction = config["chat"]["synthesizer"]["instruction"].format(questions=questions_list)
    
    # ASSEMBLE PROMPT
    prompt = config["chat"]["prompts"]["synthesizer_template"].format(
        role = config["chat"]["synthesizer"]["role"],
        instruction = instruction,
        questions = questions_list, 
        rules = config["chat"]["synthesizer"]["rules"],
        chain_of_thought = config["chat"]["synthesizer"]["chain_of_thought"],
        chain_of_verification = config["chat"]["synthesizer"]["chain_of_verification"],
        history = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in history])
    )
    
    return structured_synthesizer.invoke(prompt)