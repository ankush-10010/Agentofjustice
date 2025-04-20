from __future__ import annotations
from huggingface_hub import login
login(token="")  # You should add your token here
import os
from typing import List, Dict
from huggingface_hub import InferenceClient
import pandas as pd
import sys
import csv

class LawyerAgent:
    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "mistralai/Mistral-7B-Instruct-v0.3"):  
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []
        self.client = InferenceClient(
            model,
            token=os.getenv("HF_API_TOKEN")
        )
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

    def print_colored(self, text, text_color_code, bg_color_code=None):
        if bg_color_code:
            print(f"\033[{text_color_code};{bg_color_code}m{text}\033[0m")
        else:
            print(f"\033[{text_color_code}m{text}\033[0m")
    
    def reset_history(self):
        """Reset the agent's conversation history"""
        self.history = []

# Agent system prompts
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
You are an expert judge trained on historical case data. Your verdicts must:
1. Match patterns from similar cases in the dataset
2. Consider the strength of evidence presented
3. Follow legal standards of proof

Analysis Framework:
- Step 1: Identify key case factors (evidence type, witness credibility, etc.)
- Step 2: Compare to similar cases in the dataset
- Step 3: Apply burden of proof standards
- Step 4: Render statistically likely verdict

Output Format:
[Legal Reasoning]... 
[Dataset Comparison]...
[Verdict]: GUILTY/NOT_GUILTY
"""

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

def simulate_trial(case_id, case_text):
    """
    Run a simulated trial for a specific case
    """
    # Instantiate agents for this case
    defense_lawyer = LawyerAgent("Alex Carter (Defense)", DEFENSE_SYSTEM)
    prosecution_lawyer = LawyerAgent("Jordan Blake (Prosecution)", PROSECUTION_SYSTEM)
    defendant = LawyerAgent("John Doe (Defendant)", DEFENDENT_SYSTEM)
    plaintiff = LawyerAgent("TechCorp (Plaintiff)", PLAINTIFF_SYSTEM)
    judge = LawyerAgent("Judge Williams", JUDGE_SYSTEM)
    
    # Create jury personas
    jurry_personas = [
        {"name": "Juror 1", "system_prompt": "You are a strict and logical juror who focuses on evidence."},
        {"name": "Juror 2", "system_prompt": "You are an empathetic juror who considers human factors."},
        {"name": "Juror 3", "system_prompt": "You are a skeptical juror who questions everything."},
        # {"name": "Juror 4", "system_prompt": "You are an arrogant juror who believes in your own judgment."},
        # {"name": "Juror 5", "system_prompt": "You are a kind and forgiving juror who believes in second chances."},
        # {"name": "Juror 6", "system_prompt": "You are a strict juror who believes in harsh punishment for lawbreakers."}
    ]
    
    jury = []
    for i, persona in enumerate(jurry_personas):
        jury.append(LawyerAgent(f"Juror {i+1}", persona["system_prompt"]))

    # Create a summary of the case for the simulation
    case_summary = f"CASE ID: {case_id}\n\nSUMMARY: {case_text[:500]}..."
    
    print_header(f"CASE {case_id}")
    print(case_summary)
    
    print_header("Court Session Begins")
    opening = judge.respond(
        f"The court is now in session for Case {case_id}. "
        f"This is a case regarding {case_text[:200]}... "
        "Prosecution, you may begin with your opening statement."
    )
    judge.print_colored(f"JUDGE: {opening}", 34)  # Blue for judge

    pros_open = prosecution_lawyer.respond(
        f"Your Honor, members of the court, the facts of this case are: {case_text[:300]}... "
        "We will prove beyond reasonable doubt that the defendant "
        "is guilty of the charges brought against them."
    )
    prosecution_lawyer.print_colored(f"PROSECUTION: {pros_open}", 31)  # Red for prosecution
    
    defense_open = defense_lawyer.respond(
        f"Your Honor, regarding the case facts: {case_text[:200]}... "
        "The defense maintains the defendant's innocence. "
        "The prosecution's case rests on insufficient evidence. "
    )
    defense_lawyer.print_colored(f"DEFENSE: {defense_open}", 32)  # Green for defense
    
    print_header("Prosecution Direct Examination")
    judge_instruction = judge.respond(
        "Prosecution, you may call your first witness."
    )
    judge.print_colored(f"JUDGE: {judge_instruction}", 34)

    plaintiff_testimony = plaintiff.respond(
        f"Based on the case facts {case_text[:250]}... "
        "I can confirm that the defendant's actions caused harm as described."
    )
    plaintiff.print_colored(f"PLAINTIFF: {plaintiff_testimony}", 35)  # Purple

    print_header("Defense Cross-Examination")
    defense_q1 = defense_lawyer.respond(
        f"Considering the case details {case_text[:200]}... "
        "Isn't it true that there are alternative explanations for what happened?"
    )
    defense_lawyer.print_colored(f"DEFENSE: {defense_q1}", 32)
    
    opinions = []
    for juror in jury:
        response = juror.respond(
            f"Based on the evidence presented so far, including case details {case_text[:150]}... "
            f"and the defense's question: {defense_q1}, what is your current opinion?"
        )
        opinions.append((juror.name, response))
    
    for name, opinion in opinions:
        print(f"{name}: {opinion[:100]}...")
    
    plaintiff_response = plaintiff.respond(defense_q1)
    plaintiff.print_colored(f"PLAINTIFF: {plaintiff_response}", 35)

    print_header("Judge's Verdict")
    verdict = judge.respond(
        f"Having heard all evidence related to case {case_id} with details: {case_text[:400]}... "
        "I must now render a verdict based on the evidence presented."
    )
    judge.print_colored(f"JUDGE: {verdict}", 34)
    
    print("\n\n" + "="*80)
    print(f"END OF CASE {case_id} SIMULATION")
    print("="*80 + "\n\n")
    
    # Reset agent histories for the next case
    defense_lawyer.reset_history()
    prosecution_lawyer.reset_history()
    defendant.reset_history()
    plaintiff.reset_history()
    judge.reset_history()
    for juror in jury:
        juror.reset_history()

def main():
    # Load cases from file
    filename = r"C:\CODE\Machine_Learning\Cynaptics_Project\cases.csv"  # Update with your file path
    cases = load_cases_from_file(filename)
    
    if not cases:
        print("Failed to load cases. Exiting.")
        return
    
    print(f"Successfully loaded {len(cases)} cases.")
    
    # Interactive menu
    while True:
        print("\n" + "="*50)
        print("LEGAL CASE SIMULATION SYSTEM")
        print("="*50)
        print("1. List available case IDs")
        print("2. Run simulation for a specific case")
        print("3. Run simulation for multiple cases")
        print("4. Search for a case by ID")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # List available case IDs
            print("\nAvailable case IDs:")
            for i, case_id in enumerate(list(cases.keys())[:20], 1):
                print(f"{i}. {case_id}")
            if len(cases) > 20:
                print(f"... and {len(cases) - 20} more")
        
        elif choice == '2':
            # Run simulation for a specific case
            case_id = input("Enter case ID: ")
            if case_id in cases:
                simulate_trial(case_id, cases[case_id])
            else:
                print(f"Case ID {case_id} not found.")
        
        elif choice == '3':
            # Run simulation for multiple cases
            num_cases = int(input("How many cases to simulate? "))
            start_index = int(input("Start from index (0-based): ") or "0")
            
            case_ids = list(cases.keys())[start_index:start_index + num_cases]
            for case_id in case_ids:
                simulate_trial(case_id, cases[case_id])
                
                # Ask if user wants to continue after each case
                if case_id != case_ids[-1]:  # If not the last case
                    cont = input("Continue to next case? (y/n): ")
                    if cont.lower() != 'y':
                        break
        
        elif choice == '4':
            # Search for a case by ID
            search_term = input("Enter search term for case ID: ")
            matching_ids = [cid for cid in cases.keys() if search_term.lower() in cid.lower()]
            
            if matching_ids:
                print(f"\nFound {len(matching_ids)} matching case IDs:")
                for i, case_id in enumerate(matching_ids[:20], 1):
                    print(f"{i}. {case_id}")
                
                if len(matching_ids) > 20:
                    print(f"... and {len(matching_ids) - 20} more")
                
                # Ask if user wants to simulate one of the found cases
                sim_case = input("\nEnter number to simulate (or press Enter to skip): ")
                if sim_case.isdigit() and 1 <= int(sim_case) <= len(matching_ids):
                    case_id = matching_ids[int(sim_case) - 1]
                    simulate_trial(case_id, cases[case_id])
            else:
                print("No matching cases found.")
        
        elif choice == '5':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()