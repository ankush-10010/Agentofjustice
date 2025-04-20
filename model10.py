from __future__ import annotations
import os
from typing import List, Dict, Optional, Union
import pandas as pd
import json
from huggingface_hub import login
from huggingface_hub import InferenceClient
import sys
import csv
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

# ================== CONFIGURATION ==================
os.environ["HF_API_TOKEN"] = ""  # Replace with your token

# ================== QUOTA MANAGEMENT ==================
class QuotaTracker:
    def __init__(self, max_calls=50):
        self.max_calls = max_calls
        self.call_count = 0
    def check_quota(self):
        self.call_count += 1
        remaining = self.max_calls - self.call_count
        if remaining <= 5:
            print(f"WARNING: Only {remaining} API calls left")
        if self.call_count >= self.max_calls:
            raise RuntimeError("API quota exhausted")

quota_tracker = QuotaTracker(max_calls=50)

# ================== CORE PARTICIPANT CLASS ==================
class LegalParticipant:
    """Core class representing participants in legal proceedings"""
    
    def __init__(self,
                participant_name: str,
                participant_role: str,
                role_instructions: str,
                llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.participant_name = participant_name
        self.role = participant_role
        self.role_instructions = role_instructions.strip()
        self.conversation_history: List[Dict[str, str]] = []
        self.llm_client = InferenceClient(llm_model, token=os.getenv("HF_API_TOKEN"))
        self.llm_model = llm_model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @lru_cache(maxsize=100)
    def generate_response(self, input_message: str, **generation_params) -> str:
        """Produce a response based on the participant's role and conversation history"""
        quota_tracker.check_quota()
        
        prompt = self._format_prompt(input_message)
        
        try:
            completion = self.llm_client.text_generation(
                prompt,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                **generation_params
            )
            
            response_content = completion.strip()
            
            # Maintain conversation context
            self.conversation_history.append({"role": "user", "content": input_message})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            return response_content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"[SYSTEM ERROR: Could not generate response - {str(e)}]"
    
    def _format_prompt(self, user_msg: str) -> str:
        """Format the prompt with system instructions and history"""
        messages = [{"role": "system", "content": self.role_instructions}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_msg})

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt
    
    def reset_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# ================== LEGAL PROCEEDINGS CLASS ==================
class LegalProceedings:
    """Main class to manage and simulate legal proceedings"""
    
    def __init__(self, case_identifier: str, case_details: str):
        self.case_identifier = case_identifier
        self.case_details = case_details
        self.proceedings_record = []
        self.participants = {}
        self.testifying_witnesses = []
        self.final_decision = None
        
        # Set up required participants
        self._setup_required_participants()
    
    def _setup_required_participants(self):
        """Initialize essential participants for the legal proceedings"""
        self.participants["presiding_judge"] = LegalParticipant(
            participant_name="Justice Williams",
            participant_role="judicial_officer",
            role_instructions=JUDICIAL_INSTRUCTIONS
        )
        
        self.participants["defense_attorney"] = LegalParticipant(
            participant_name="Taylor Reed",
            participant_role="defense_counsel",
            role_instructions=DEFENSE_COUNSEL_INSTRUCTIONS
        )
        
        self.participants["prosecuting_attorney"] = LegalParticipant(
            participant_name="Casey Jordan",
            participant_role="prosecutor",
            role_instructions=PROSECUTION_INSTRUCTIONS
        )
        
        self.participants["accused_party"] = LegalParticipant(
            participant_name="Respondent",
            participant_role="defendant",
            role_instructions=RESPONDENT_INSTRUCTIONS
        )
        
        self.participants["complaining_party"] = LegalParticipant(
            participant_name="Petitioner",
            participant_role="plaintiff",
            role_instructions=PETITIONER_INSTRUCTIONS
        )
    
    def _record_interaction(self, participant_role: str, statement: str):
        """Document an interaction in the proceedings record"""
        speaker_identification = self.participants[participant_role].participant_name
        record_entry = {
            "participant_type": participant_role,
            "participant_name": speaker_identification,
            "statement": statement
        }
        self.proceedings_record.append(record_entry)
        print(f"{speaker_identification.upper()} ({participant_role}):\n{statement}\n")
    
    def add_testifying_witness(self, witness_name: str, witness_background: str) -> str:
        """Introduce a new witness to the proceedings"""
        witness_identifier = f"testifying_witness_{len(self.testifying_witnesses) + 1}"
        witness_guidelines = WITNESS_GUIDELINES.format(
            name=witness_name,
            background=witness_background
        )
        
        self.participants[witness_identifier] = LegalParticipant(
            participant_name=witness_name,
            participant_role="witness",
            role_instructions=witness_guidelines
        )
        
        self.testifying_witnesses.append(witness_identifier)
        return witness_identifier
    
    def conduct_opening_phase(self):
        """Execute the opening statements phase"""
        print("==== PROCEEDINGS PHASE 1: INITIAL STATEMENTS ====\n")
        
        # Judicial officer commences proceedings
        commencement = self.participants["presiding_judge"].generate_response(
            f"Commence proceedings and invite opening statements for case: {self.case_details}"
        )
        self._record_interaction("presiding_judge", commencement)
        
        # Prosecution presents initial statement
        prosecution_opening = self.participants["prosecuting_attorney"].generate_response(
            f"Present initial statement regarding: {self.case_details}"
        )
        self._record_interaction("prosecuting_attorney", prosecution_opening)
        
        # Defense presents counter-statement
        defense_opening = self.participants["defense_attorney"].generate_response(
            f"Respond to prosecution's opening with counter-statement: {prosecution_opening}"
        )
        self._record_interaction("defense_attorney", defense_opening)
    
    def conduct_evidence_presentation(self):
        """Execute the evidence presentation phase"""
        print("==== PROCEEDINGS PHASE 2: EVIDENTIARY HEARING ====\n")
        
        # Judicial officer invites evidence presentation
        evidence_invitation = self.participants["presiding_judge"].generate_response(
            "Invite parties to present evidentiary materials"
        )
        self._record_interaction("presiding_judge", evidence_invitation)
        
        # Determine appropriate witnesses
        witness_recommendation = self.participants["prosecuting_attorney"].generate_response(
            f"Suggest key witnesses for case: {self.case_details}"
        )
        self._record_interaction("prosecuting_attorney", witness_recommendation)
        
        # Establish witnesses for both sides
        petitioner_witness = self.add_testifying_witness(
            "Dr. Morgan Taylor", 
            "Subject matter expert with case-specific knowledge"
        )
        
        respondent_witness = self.add_testifying_witness(
            "Alex Jordan",
            "Reputation witness familiar with respondent's character"
        )
        
        # Prosecution examines primary witness
        direct_examination = self.participants["prosecuting_attorney"].generate_response(
            f"Examine Dr. Morgan Taylor regarding: {self.case_details}"
        )
        self._record_interaction("prosecuting_attorney", direct_examination)
        
        # Witness testimony
        witness_testimony = self.participants[petitioner_witness].generate_response(
            f"Provide testimony regarding: {direct_examination}"
        )
        self._record_interaction(petitioner_witness, witness_testimony)
        
        # Defense cross-examination
        cross_examination = self.participants["defense_attorney"].generate_response(
            f"Cross-examine witness based on: {witness_testimony}"
        )
        self._record_interaction("defense_attorney", cross_examination)
        
        # Witness response to cross
        cross_response = self.participants[petitioner_witness].generate_response(
            f"Respond to cross-examination: {cross_examination}"
        )
        self._record_interaction(petitioner_witness, cross_response)
        
        # Defense presents their witness
        defense_examination = self.participants["defense_attorney"].generate_response(
            f"Examine Alex Jordan regarding: {self.case_details}"
        )
        self._record_interaction("defense_attorney", defense_examination)
        
        # Defense witness testimony
        defense_testimony = self.participants[respondent_witness].generate_response(
            f"Provide testimony regarding: {defense_examination}"
        )
        self._record_interaction(respondent_witness, defense_testimony)
        
        # Prosecution cross-examination
        prosecution_cross = self.participants["prosecuting_attorney"].generate_response(
            f"Cross-examine based on testimony: {defense_testimony}"
        )
        self._record_interaction("prosecuting_attorney", prosecution_cross)
        
        # Defense witness response
        defense_cross_response = self.participants[respondent_witness].generate_response(
            f"Respond to cross-examination: {prosecution_cross}"
        )
        self._record_interaction(respondent_witness, defense_cross_response)
    
    def conduct_closing_phase(self):
        """Execute the closing arguments phase"""
        print("==== PROCEEDINGS PHASE 3: FINAL ARGUMENTS ====\n")
        
        # Judicial officer invites closing
        closing_invitation = self.participants["presiding_judge"].generate_response(
            "Invite parties to present concluding arguments"
        )
        self._record_interaction("presiding_judge", closing_invitation)
        
        # Prosecution closing
        case_summary = self._generate_case_summary()
        prosecution_closing = self.participants["prosecuting_attorney"].generate_response(
            f"Present concluding arguments summarizing evidence. Summary: {case_summary}"
        )
        self._record_interaction("prosecuting_attorney", prosecution_closing)
        
        # Defense closing
        defense_closing = self.participants["defense_attorney"].generate_response(
            f"Present counter-arguments to prosecution's closing: {prosecution_closing}"
        )
        self._record_interaction("defense_attorney", defense_closing)
    
    def conduct_adjudication(self):
        """Execute the final adjudication phase"""
        print("==== PROCEEDINGS PHASE 4: JUDICIAL DETERMINATION ====\n")
        
        # Generate case summary
        case_summary = self._generate_case_summary()
        
        # Judicial determination with structured output
        determination_prompt = f"""
        [CASE DETAILS]
        {self.case_details}
        
        [PROCEEDINGS SUMMARY]
        {case_summary}
        
        You MUST follow this structured analysis:
        1. Evidence Analysis: Categorize and weigh ALL evidence
        2. Legal Standard Application: Apply proper burden of proof
        3. Precedent Comparison: Cite relevant precedents
        4. Final Verdict: Conclude with either APPROVED or REJECTED
        
        If evidence is exactly balanced, the verdict MUST be REJECTED.
        """
        
        judicial_determination = self.participants["presiding_judge"].generate_response(determination_prompt)
        self._record_interaction("presiding_judge", judicial_determination)
        
        # Extract and enforce definitive determination
        if "APPROVED" in judicial_determination.upper() and "REJECTED" not in judicial_determination.upper():
            self.final_decision = 1
        else:
            self.final_decision = 0
        
        return self.final_decision
    
    def _generate_case_summary(self) -> str:
        """Create condensed summary of proceedings"""
        summary = "Case Proceedings Summary:\n"
        for entry in self.proceedings_record[-10:]:
            summary += f"- {entry['participant_name']} ({entry['participant_type']}): {entry['statement'][:100]}...\n"
        return summary
    
    def execute_full_proceedings(self) -> int:
        """Execute complete legal proceedings"""
        print(f"\n===== COMMENCING PROCEEDINGS FOR CASE {self.case_identifier} =====\n")
        print(f"CASE DETAILS: {self.case_details}\n")
        
        # Execute all phases
        self.conduct_opening_phase()
        self.conduct_evidence_presentation()
        self.conduct_closing_phase()
        determination = self.conduct_adjudication()
        
        print(f"\n===== PROCEEDINGS CONCLUDED: OUTCOME {'APPROVED' if determination == 1 else 'REJECTED'} =====\n")
        return determination

# ================== ROLE INSTRUCTIONS ==================
JUDICIAL_INSTRUCTIONS = """
As *Justice Williams* presiding over these proceedings:
Responsibilities:
• Maintain proper courtroom decorum
• Ensure adherence to legal protocols
• Rule on procedural matters
• Deliver impartial judgment based on presented evidence
Communication Style:
• Authoritative yet fair
• Precise legal terminology
• Equal treatment of all parties
Ethical Standards:
• Base decisions solely on legal merits
• Exhibit no partiality
• Guarantee fair opportunity for all parties

Final determination must be either APPROVED (petitioner) or REJECTED (respondent).
"""

DEFENSE_COUNSEL_INSTRUCTIONS = """
As *Taylor Reed*, defense counsel:
Objectives:
• Protect respondent's legal rights
• Establish reasonable doubt
• Maintain professional decorum
Communication Approach:
• Persuasive, precedent-based
• Cite relevant cases briefly (e.g., Roe v. Wade (1973))
Professional Ethics:
• Avoid factual misrepresentation
• Admit knowledge gaps when necessary
• Provide vigorous defense within ethical bounds
"""

PROSECUTION_INSTRUCTIONS = """
As *Casey Jordan*, prosecuting attorney:
Objectives:
• Present compelling case
• Structure logical evidentiary presentation
• Anticipate defense arguments
Communication Approach:
• Clear, convincing narrative
• Confident but measured tone
Professional Ethics:
• Prioritize justice over victory
• Acknowledge valid counterpoints
• Present evidence fairly while advocating your position
"""

RESPONDENT_INSTRUCTIONS = """
As the respondent:
Objectives:
• Provide truthful responses
• Support defense strategy
• Maintain appropriate demeanor
Communication Style:
• Respectful to the court
• Clear and credible responses
• Appropriate emotional expression
Ethical Standards:
• Avoid false statements
• Present favorable aspects truthfully
• Remain composed during questioning
"""

PETITIONER_INSTRUCTIONS = """
As the petitioner:
Objectives:
• Clearly present grievance
• Support claims with evidence
• Strengthen case through testimony
Communication Style:
• Sincere and consistent
• Emphasize experienced harm
• Maintain factual accuracy
Ethical Standards:
• Present facts truthfully
• Avoid exaggeration
• Demonstrate entitlement to remedy
"""

WITNESS_GUIDELINES = """
As {name}, testifying witness:
Background:
{background}

Objectives:
• Provide accurate testimony
• Respond truthfully within knowledge
• Maintain credibility
Communication Style:
• Fact-based responses
• Appropriate to expertise
• Distinguish knowledge from speculation
Ethical Standards:
• Testify truthfully
• Avoid exceeding knowledge boundaries
• Remain composed during examination
"""

# ================== CASE PROCESSING ==================
def analyze_legal_cases(input_file: str):
    """Analyze legal cases from input file"""
    case_data = pd.read_csv(input_file)
    
    determinations = []
    
    for _, record in case_data.iterrows():
        case_id = record['id']
        case_content = record['text']
        
        if not isinstance(case_content, str) or len(case_content) < 10:
            determinations.append((case_id, 0))
            continue
            
        try:
            proceedings = LegalProceedings(case_id, case_content)
            outcome = proceedings.execute_full_proceedings()
            determinations.append((case_id, outcome))
        except Exception as error:
            print(f"Processing error for case {case_id}: {error}")
            determinations.append((case_id, 0))
    
    results = pd.DataFrame(determinations, columns=['ID', 'DETERMINATION'])
    return results

def load_cases_from_file(filename):
    """Load legal cases from a CSV file"""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = max_size // 10
    
    cases = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'id' in row and 'text' in row:
                    cases[row['id']] = row['text']
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return cases

def generate_report(results):
    """Generate a summary report of all case verdicts"""
    with open("verdict_summary.txt", "w") as f:
        f.write("LEGAL PROCEEDINGS SUMMARY REPORT\n")
        f.write("="*50 + "\n\n")
        
        for case_id, determination in results:
            f.write(f"Case {case_id}: {'APPROVED' if determination == 1 else 'REJECTED'}\n")
        
        approved = sum(1 for _, d in results if d == 1)
        rejected = len(results) - approved
        f.write(f"\nTotal Cases: {len(results)}\n")
        f.write(f"Approved: {approved} ({approved/len(results)*100:.1f}%)\n")
        f.write(f"Rejected: {rejected} ({rejected/len(results)*100:.1f}%)\n")

if __name__ == "__main__":
    # Demonstration case
    sample_case = """
    The petitioner claims the respondent violated contractual terms by establishing
    a competing enterprise within prohibited timeframe. The agreement stipulated
    12-month restriction. Respondent argues the clause is unreasonably extensive
    geographically. Documentation indicates respondent launched similar operation
    within 13 kilometers of petitioner's establishment.
    """
    
    demo_proceedings = LegalProceedings("demo_1", sample_case)
    case_outcome = demo_proceedings.execute_full_proceedings()
    print(f"Final determination: {'APPROVED' if case_outcome == 1 else 'REJECTED'}")
    
    # Process actual case file
    case_results = analyze_legal_cases("legal_cases.csv")
    case_results.to_csv("legal_determinations.csv", index=False)
    generate_report(case_results)