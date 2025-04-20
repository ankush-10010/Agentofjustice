from __future__ import annotations
from huggingface_hub import login
login(token="")  # Replace with your token
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
                 model: str = "mistralai/Mistral-7B-Instruct-v0.3"):  
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

Your CORE RESPONSIBILITY is to render accurate verdicts based on:
1. Concrete evidence presented in the case
2. Legal precedent and applicable statutes
3. Proper application of the burden of proof ("beyond reasonable doubt" for criminal cases, "preponderance of evidence" for civil)

ANALYTICAL FRAMEWORK (you must follow each step):
1. Evidence Analysis: Categorize and weigh ALL evidence (documentary, testimonial, circumstantial)
2. Legal Standard Application: Apply the specific burden of proof required for this case type
3. Precedent Comparison: Identify and cite at least 2 relevant precedents
4. Reasoned Conclusion: Synthesize findings into a clear verdict

YOUR OUTPUT MUST FOLLOW THIS EXACT FORMAT:
[EVIDENCE ANALYSIS]
- Key evidence item 1: [weight and credibility]
- Key evidence item 2: [weight and credibility]
...

[LEGAL STANDARDS APPLIED]
- Burden of proof: [standard for this case]
- Key legal test: [specific test from relevant precedent]

[PRECEDENT COMPARISON]
- Case 1 [name, year]: [brief description and relevance]
- Case 2 [name, year]: [brief description and relevance]

[FINAL VERDICT]: GUILTY/NOT_GUILTY

The verdict MUST be a logical consequence of the analysis provided and not defaulted.
"""

# ================== TRIAL SIMULATION ==================
def simulate_trial(case_id, case_text):
    # Initialize agents
    defense = LawyerAgent("Alex Carter (Defense)", DEFENSE_SYSTEM)
    prosecution = LawyerAgent("Jordan Blake (Prosecution)", PROSECUTION_SYSTEM)
    judge = LawyerAgent("Judge Williams", JUDGE_SYSTEM)

    # Shortened trial procedure to conserve quota
    print_header(f"CASE {case_id}")
    
    # 1. Opening statements
    prosecution_open = prosecution.respond(
        f"Your Honor, we will prove the defendant is guilty of: {case_text[:300]}"
    )
    defense_open = defense.respond(
        f"We assert innocence because: {case_text[:200]} lacks concrete evidence"
    )

    # 2. Critical testimony (prioritize)
    testimony = (
        f"Key evidence includes: {case_text[:400]}\n"
        f"Defendant claims: {case_text[:200]}"
    )

    # 3. JUDGE'S VERDICT (priority section)
    verdict_template = JUDGE_SYSTEM.format(
        evidence_summary=testimony[:500],
        standard="beyond reasonable doubt",
        verdict="GUILTY" if len(case_text) > 1000 else "NOT GUILTY"  # Simple heuristic
    )
    
    verdict = judge.respond(
        f"Based on: {testimony[:300]}\n"
        "Render verdict using required format."
    )
    
    # Force verdict if not properly formatted
    if "FINAL VERDICT:" not in verdict:
        verdict += "\nFINAL VERDICT: NOT GUILTY (default due to insufficient evidence)"
    
    # Highlight verdict
    if "GUILTY" in verdict:
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

# ================== MAIN EXECUTION ==================
def main():
    cases = load_cases_from_file("cases.csv")  # Your file path
    
    for i, (case_id, case_text) in enumerate(cases.items()):
        if i >= 2:  # Limit to 3 cases to avoid quota issues
            print("Stopping early to conserve quota")
            break
            
        verdict = simulate_trial(case_id, case_text)
        
        with open("verdicts.log", "a") as f:
            f.write(f"{case_id}: {verdict}\n")

if __name__ == "__main__":
    main()