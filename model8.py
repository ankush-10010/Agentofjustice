from __future__ import annotations
from huggingface_hub import login
import os
os.environ["HF_API_TOKEN"] = ""   # Replace with your token
import os
from typing import List, Dict
from huggingface_hub import InferenceClient
import pandas as pd
import sys
import csv
from functools import lru_cache
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# ================== QUOTA MANAGEMENT ==================
class HFQuotaTracker:
    def __init__(self, max_calls=50):
        self.max_calls = max_calls
        self.call_count = 0
    
    def check_quota(self):
        self.call_count += 1
        remaining = self.max_calls - self.call_count
        if remaining <= 5:
            print(f"WARNING: Only {remaining} API calls left")
        if self.call_count >= self.max_calls:
            raise RuntimeError("HF API quota exhausted")

quota_tracker = HFQuotaTracker(max_calls=50)  # Adjust based on your plan

# ================== CACHED CLIENT ==================
class CachedInferenceClient:
    def __init__(self, model_name, token):
        self.client = InferenceClient(model_name, token=token)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @lru_cache(maxsize=100)
    def text_generation(self, prompt, **kwargs):
        quota_tracker.check_quota()
        return self.client.text_generation(prompt, **kwargs)

# ================== AGENT CLASS ==================
class LawyerAgent:
    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "llama3-70b-8192"):  
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []
        self.client = CachedInferenceClient(model, token=os.getenv("HF_API_TOKEN"))
    
    def _format_prompt(self, user_msg: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"  
        prompt += "<|assistant|>\n"
        return prompt

    def respond(self, user_msg: str, **gen_kwargs) -> str:
        try:
            prompt = self._format_prompt(user_msg)
            completion = self.client.text_generation(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                stream=False,
                **gen_kwargs
            )
            answer = completion.strip()
            self.history.append({"role": "user", "content": user_msg})
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            return f"[ERROR: {str(e)} - Using fallback response]"

    def print_colored(self, text, text_color_code, bg_color_code=None):
        if bg_color_code:
            print(f"\033[{text_color_code};{bg_color_code}m{text}\033[0m")
        else:
            print(f"\033[{text_color_code}m{text}\033[0m")
    
    def reset_history(self):
        self.history = []

# ================== SYSTEM PROMPTS ==================
DEFENSE_SYSTEM = """
You are **Alex Carter**, lead *defense counsel*.
Goals:
• Protect the constitutional rights of the defendant.
• Raise reasonable doubt by pointing out missing evidence or alternative explanations.
• Be respectful to the Court and to opposing counsel.
Style:
• Crisp, persuasive, grounded in precedent and facts provided.
• When citing precedent: give short case name + year (e.g., *Miranda v. Arizona* (1966)).
Ethics:
• Do not fabricate evidence; admit uncertainty when required.
"""

PROSECUTION_SYSTEM = """
You are **Jordan Blake**, *Assistant District Attorney* for the State.
Goals:
• Present the strongest good‑faith case against the accused.
• Lay out facts logically, citing exhibits or witness statements when available.
• Anticipate and rebut common defense arguments.
Style:
• Formal but plain English; persuasive, with confident tone.
Ethics:
• Duty is to justice, not merely to win. Concede points when ethically required.
"""

DEFENDENT_SYSTEM = """
You are the **Defendant** in this case.  
Goals:  
• Assert your innocence or minimize culpability.  
• Provide truthful testimony but avoid self-incrimination.  
• Trust your defense counsel to guide legal strategy.  

Style:  
• Direct, factual, and emotionally restrained (e.g., "I was not there that night").  
• Defer to your lawyer for legal arguments (*"I'll let my attorney address that"*).  

Ethics:  
• Never lie under oath. Invoke the Fifth Amendment if necessary.  
• Follow courtroom decorum (address the judge as *"Your Honor"*).  
"""

PLAINTIFF_SYSTEM = """
You are the **Plaintiff** (or victim/complainant in criminal cases).  
Goals:  
• Seek justice or compensation for alleged harm.  
• Clearly state facts supporting your claim.  

Style:  
• Firm but respectful (e.g., *"The defendant's actions caused me significant harm"*).  
• Avoid emotional outbursts; rely on evidence.  

Ethics:  
• Do not exaggerate claims. Correct inaccuracies if challenged.  
• Cooperate with cross-examination.  
"""

JUDGE_SYSTEM = """
You are **Judge Williams**, an experienced legal expert with 30+ years on the bench.

Your CORE RESPONSIBILITY is to render definitive verdicts based on:
1. Concrete evidence presented in the case
2. Legal precedent and applicable statutes
3. Proper application of the burden of proof ("beyond reasonable doubt" for criminal cases, "preponderance of evidence" for civil)

ANALYTICAL FRAMEWORK (you must follow each step):
1. Evidence Analysis: Categorize and weigh ALL evidence (documentary, testimonial, circumstantial)
2. Legal Standard Application: Apply the specific burden of proof required for this case type
3. Precedent Comparison: Identify and cite at least 2 relevant precedents
4. Reasoned Conclusion: Synthesize findings into a clear verdict

YOU MUST RENDER A DEFINITIVE VERDICT - "UNDETERMINED" IS NOT AN OPTION.
When evidence is exactly balanced, the verdict MUST favor the defendant.

YOUR OUTPUT MUST FOLLOW THIS EXACT FORMAT:
[EVIDENCE ANALYSIS]
• Key evidence item 1: [weight and credibility]
• Key evidence item 2: [weight and credibility]
...

[LEGAL STANDARDS APPLIED]
• Burden of proof: [standard for this case]
• Key legal test: [specific test from relevant precedent]

[PRECEDENT COMPARISON]
• Case 1 [name, year]: [brief description and relevance]
• Case 2 [name, year]: [brief description and relevance]

[FINAL VERDICT]: GUILTY/NOT_GUILTY
"""
# ================== LEGAL KNOWLEDGE BASE ==================
LEGAL_KNOWLEDGE_BASE = {
    "criminal": {
        "burden": "beyond reasonable doubt",
        "key_precedents": [
            {"name": "In re Winship (1970)", "principle": "Due process requires proof beyond reasonable doubt in criminal cases"},
            {"name": "Coffin v. United States (1895)", "principle": "Presumption of innocence is a basic component of a fair trial"}
        ]
    },
    "civil": {
        "burden": "preponderance of evidence",
        "key_precedents": [
            {"name": "Addington v. Texas (1979)", "principle": "Civil cases generally require preponderance of evidence standard"},
            {"name": "Grogan v. Garner (1991)", "principle": "Confirms preponderance standard for civil fraud cases"}
        ]
    }
}

def get_legal_context(case_elements):
    """Retrieve relevant legal information based on case elements"""
    case_type = case_elements.get("case_type", "civil")  # Default to civil if unknown
    legal_context = LEGAL_KNOWLEDGE_BASE.get(case_type, LEGAL_KNOWLEDGE_BASE["civil"])
    
    # Add case-specific precedents based on charges/claims
    additional_precedents = []
    charges = case_elements.get("charges", [])
    
    for charge in charges:
        if "fraud" in charge.lower():
            additional_precedents.append({
                "name": "United States v. Dial (1985)",
                "principle": "Establishes standards for mail/wire fraud cases"
            })
        elif "negligence" in charge.lower():
            additional_precedents.append({
                "name": "Palsgraf v. Long Island Railroad Co. (1928)",
                "principle": "Establishes proximate cause test for negligence"
            })
    
    return {**legal_context, "additional_precedents": additional_precedents}

# ================== CASE PROCESSING ==================
def preprocess_case(case_text):
    """Extract key elements from case text to help with judgment"""
    # Define case structure components
    sections = {
        "case_type": None,
        "parties": [],
        "charges": [],
        "key_evidence": [],
        "witness_testimony": [],
        "legal_context": []
    }
    
    # Extract case type (criminal/civil)
    if "criminal" in case_text.lower():
        sections["case_type"] = "criminal"
    elif "civil" in case_text.lower():
        sections["case_type"] = "civil"
    
    # Simple extraction of parties (could be improved with NER)
    if "v." in case_text:
        parties = case_text.split("v.")[0:2]
        sections["parties"] = [p.strip() for p in parties]
    
    # Extract any charges mentioned
    common_charges = ["theft", "fraud", "murder", "assault", "negligence"]
    for charge in common_charges:
        if charge in case_text.lower():
            sections["charges"].append(charge)
    
    # Extract evidence mentions
    evidence_indicators = ["exhibit", "evidence", "witness", "testimony", "document"]
    for indicator in evidence_indicators:
        if indicator in case_text.lower():
            # Find the sentence containing this evidence
            sentences = case_text.split(". ")
            for sent in sentences:
                if indicator in sent.lower():
                    sections["key_evidence"].append(sent.strip())
    
    return sections

# ================== VERDICT VALIDATION ==================
def validate_verdict(verdict):
    """Check if the verdict contains proper legal reasoning and is consistent"""
    required_sections = [
        "[EVIDENCE ANALYSIS]", 
        "[LEGAL STANDARDS APPLIED]",
        "[PRECEDENT COMPARISON]", 
        "[FINAL VERDICT]"
    ]
    
    validation_issues = []
    
    # Check for required sections
    for section in required_sections:
        if section not in verdict:
            validation_issues.append(f"Missing {section} section")
    
    # Check consistency between reasoning and verdict
    evidence_section = verdict.split("[EVIDENCE ANALYSIS]")[1].split("[")[0] if "[EVIDENCE ANALYSIS]" in verdict else ""
    verdict_section = verdict.split("[FINAL VERDICT]")[1] if "[FINAL VERDICT]" in verdict else ""
    
    evidence_strength = 0
    if "compelling" in evidence_section.lower() or "strong" in evidence_section.lower():
        evidence_strength = 2
    elif "sufficient" in evidence_section.lower():
        evidence_strength = 1
    elif "weak" in evidence_section.lower() or "insufficient" in evidence_section.lower():
        evidence_strength = -1
    
    if evidence_strength > 0 and "NOT_GUILTY" in verdict_section:
        validation_issues.append("Inconsistency: Strong evidence but Not Guilty verdict")
    elif evidence_strength < 0 and "GUILTY" in verdict_section:
        validation_issues.append("Inconsistency: Weak evidence but Guilty verdict")
    
    return validation_issues

# ================== TRIAL SIMULATION ==================
def simulate_trial(case_id, case_text, judge=None):
    # Initialize agents
    defense = LawyerAgent("Alex Carter (Defense)", DEFENSE_SYSTEM)
    prosecution = LawyerAgent("Jordan Blake (Prosecution)", PROSECUTION_SYSTEM)
    
    if judge is None:
        judge = LawyerAgent("Judge Williams", JUDGE_SYSTEM)
    
    # Process case data
    case_elements = preprocess_case(case_text)
    
    # 1. Opening statements
    case_summary = f"""
    Case Type: {case_elements['case_type'] or 'Unspecified'}
    Involved Parties: {', '.join(case_elements['parties']) if case_elements['parties'] else 'Unspecified'}
    Charges/Claims: {', '.join(case_elements['charges']) if case_elements['charges'] else 'Unspecified'}
    Key Evidence Available: {len(case_elements['key_evidence'])} items
    """
    
    prosecution_open = prosecution.respond(
        f"Your Honor, the State will prove the defendant's guilt beyond reasonable doubt. Case overview: {case_summary}"
    )
    
    defense_open = defense.respond(
        f"Your Honor, we will establish reasonable doubt. Case overview: {case_summary}"
    )
    
    # 2. Present evidence
    evidence_presentation = ""
    for i, evidence in enumerate(case_elements['key_evidence']):
        evidence_presentation += f"Evidence Item #{i+1}: {evidence}\n"
    
    prosecution_evidence = prosecution.respond(
        f"I present the following evidence:\n{evidence_presentation}"
    )
    
    defense_rebuttal = defense.respond(
        f"I challenge the prosecution's evidence as follows:\n{evidence_presentation}"
    )
    
    # 3. Judge's verdict with enforcement of definitive ruling
    verdict_prompt = f"""
    Complete case file for judgment:
    
    CASE SUMMARY:
    {case_summary}
    
    FULL CASE TEXT:
    {case_text[:1000]}
    
    PROSECUTION ARGUMENTS:
    {prosecution_open}
    {prosecution_evidence}
    
    DEFENSE ARGUMENTS:
    {defense_open}
    {defense_rebuttal}
    
    You MUST render a definitive verdict of either GUILTY or NOT_GUILTY.
    If the evidence is exactly balanced, the verdict MUST be NOT_GUILTY.
    """
    
    verdict = judge.respond(verdict_prompt)
    
    # Enforce definitive verdict
    if "GUILTY" in verdict and "NOT_GUILTY" not in verdict:
        final_verdict = "GUILTY"
    else:
        final_verdict = "NOT_GUILTY"
    
    # Ensure proper formatting
    if "[FINAL VERDICT]:" not in verdict:
        verdict = f"{verdict}\n[FINAL VERDICT]: {final_verdict}"
    else:
        # Replace any existing verdict with the enforced one
        verdict = verdict.split("[FINAL VERDICT]:")[0] + f"[FINAL VERDICT]: {final_verdict}"
    
    # Highlight verdict
    if final_verdict == "GUILTY":
        judge.print_colored(verdict, 31, 47)  # Red on white
    else:
        judge.print_colored(verdict, 32, 47)  # Green on white
    
    return verdict


def load_cases_from_file(filename):
    """
    Load legal cases from a CSV file into a dictionary
    where the keys are case IDs and values are the case text
    """
    # Increase the CSV field size limit
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
            # Read the first line to check format
            first_line = file.readline().strip()
            file.seek(0)  # Go back to start of file
            
            if first_line.startswith(',id,text'):
                # Use DictReader for properly formatted CSV
                reader = csv.DictReader(file)
                for row in reader:
                    if 'id' in row and 'text' in row:
                        cases[row['id']] = row['text']
            else:
                # Custom parsing for malformed CSV
                content = file.read()
                rows = content.split('\n')
                current_id = None
                current_text = ""
                
                for row in rows:
                    if row.strip() == "":
                        continue
                    
                    parts = row.split(',', 2)
                    if len(parts) >= 3 and parts[1].strip():
                        # Save previous case if we have one
                        if current_id is not None:
                            cases[current_id] = current_text
                        
                        # Start new case
                        index, case_id, text = parts
                        current_id = case_id.strip('"')
                        current_text = text
                    elif current_id is not None:
                        # Continue previous case
                        current_text += " " + row
                
                # Save the last case
                if current_id is not None:
                    cases[current_id] = current_text
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return cases

def print_header(title):
    print("\n" + "="*50)
    print(f"== {title.upper()} ==")
    print("="*50 + "\n")

# ================== REPORTING ==================
def generate_report(results):
    """Generate a summary report of all case verdicts (appends to existing file)"""
    guilty_count = sum(1 for r in results if r["verdict"] == "GUILTY")
    not_guilty_count = sum(1 for r in results if r["verdict"] == "NOT_GUILTY")
    validation_issues_count = sum(1 for r in results if r["validation_issues"])
    
    # Get current timestamp for the report section
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("verdict_summary.txt", "a") as f:  # Changed to append mode ('a')
        # Add separator and timestamp for new session
        f.write("\n\n" + "="*50 + "\n")
        f.write(f"NEW VERDICT SESSION - {timestamp}\n")
        f.write("="*50 + "\n\n")
        
        # Summary statistics
        f.write(f"Total Cases Analyzed in This Session: {len(results)}\n")
        f.write(f"Guilty Verdicts: {guilty_count} ({guilty_count/len(results)*100:.1f}%)\n")
        f.write(f"Not Guilty Verdicts: {not_guilty_count} ({not_guilty_count/len(results)*100:.1f}%)\n")
        f.write(f"Verdicts with Validation Issues: {validation_issues_count} ({validation_issues_count/len(results)*100:.1f}%)\n\n")
        
        # Case details
        f.write("CASE DETAILS:\n")
        for result in results:
            f.write(f"- Case {result['case_id']}: {result['verdict']}\n")
            if result["validation_issues"]:
                f.write(f"  • Issues: {', '.join(result['validation_issues'])}\n")
            if "full_verdict" in result:
                f.write(f"  • Key Excerpt: {result['full_verdict'][:200]}...\n")
        
        f.write("\nEND OF SESSION REPORT")
    
    print(f"\nReport appended to verdict_summary.txt (timestamp: {timestamp})")
# ================== MAIN EXECUTION ==================
def main():
    cases = load_cases_from_file("cases.csv")
    
    if not cases:
        print("No cases loaded. Please check your cases.csv file.")
        return
    
    # Display available cases
    print("\nAvailable Cases:")
    for case_id in cases.keys():
        print(f"- {case_id}")
    
    # Let user select cases
    selected_ids = input("\nEnter case IDs to judge (comma-separated): ").strip().split(',')
    selected_ids = [id.strip() for id in selected_ids if id.strip()]
    
    # Validate selection
    valid_cases = []
    for case_id in selected_ids:
        if case_id in cases:
            valid_cases.append((case_id, cases[case_id]))
        else:
            print(f"Warning: Case ID '{case_id}' not found - skipping")
    
    if not valid_cases:
        print("No valid cases selected. Exiting.")
        return
    
    # Results tracking
    results = []
    
    for case_id, case_text in valid_cases:
        print_header(f"CASE {case_id}")
        
        # Process case elements
        case_elements = preprocess_case(case_text)
        
        # Get relevant legal context
        legal_context = get_legal_context(case_elements)
        
        # Enhance judge with legal context
        judge = LawyerAgent(
            "Judge Williams", 
            JUDGE_SYSTEM + "\n\nLegal context for this case: " + str(legal_context)
        )
        
        # Simulate trial
        verdict = simulate_trial(case_id, case_text, judge)
        
        # Validate verdict
        validation_issues = validate_verdict(verdict)
        if validation_issues:
            print("WARNING: Verdict validation issues:")
            for issue in validation_issues:
                print(f"- {issue}")
        
        # Store results
        results.append({
            "case_id": case_id,
            "verdict": "GUILTY" if "GUILTY" in verdict and "NOT_GUILTY" not in verdict else "NOT_GUILTY",
            "validation_issues": validation_issues,
            "full_verdict": verdict
        })
        
        # Log to file
        with open("verdicts.log", "a") as f:
            f.write(f"{case_id}: {verdict}\n")
            if validation_issues:
                f.write(f"VALIDATION ISSUES: {validation_issues}\n")
            f.write("="*50 + "\n")
    
    # Generate summary report
    generate_report(results)
    print("\nReport generated as 'verdict_summary.txt'")

if __name__ == "__main__":
    main()