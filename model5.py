from __future__ import annotations
from huggingface_hub import login
login(token="")
import os
from typing import List, Dict
from huggingface_hub import InferenceClient
import pandas as pd
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

# class JurryAgent:
    def __init__(self,name:str,
                 system_prompt: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []
        self.client = InferenceClient(
            model,
            token=os.getenv("HF_API_TOKEN")
        )

DEFENSE_SYSTEM = """
You are **Alex Carter**, lead *defense counsel*.
Goals:
• Protect the constitutional rights of the defendant.
• Raise reasonable doubt by pointing out missing evidence or alternative explanations.
• Be respectful to the Court and to opposing counsel.
Style:
• Crisp, persuasive, grounded in precedent and facts provided.
• When citing precedent: give short case name + year (e.g., *Miranda v. Arizona* (1966)).
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
• Defer to your lawyer for legal arguments (*"I’ll let my attorney address that"*).  

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
• Firm but respectful (e.g., *"The defendant’s actions caused me significant harm"*).  
• Avoid emotional outbursts; rely on evidence.  

Ethics:  
• Do not exaggerate claims. Correct inaccuracies if challenged.  
• Cooperate with cross-examination.  
"""



"""
changed the judge system prompt
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
#idea to create multiple jurry personas by giving an local llm command itertively to create multiple jurry personas
jurry_personas=[
    {"name": "Jurry 1", "system_prompt": "strict and logical",},
    {"name": "Jurry 2", "system_prompt": "empathetic",},
    {"name": "Jurry 3", "system_prompt": "skeptical",},
    {"name": "Jurry 4", "system_prompt": "arrogant",},
    {"name": "Jurry 5", "system_prompt": "kind and forgiving",},
    {"name": "Jurry 6", "system_prompt": "evil and cunning",}        
]

for i in range(1,len(jurry_personas)+1):
    jurry_personas[i-1] = LawyerAgent(jurry_personas[i-1]["name"], jurry_personas[i-1]["system_prompt"])
#now we have 6 jurry personas just like lawyer agents
 #creating instance of all the agents in the courtroom
defense_lawyer = LawyerAgent("Alex Carter (Defense)", DEFENSE_SYSTEM)
prosecution_lawyer = LawyerAgent("Jordan Blake (Prosecution)", PROSECUTION_SYSTEM)
defendant = LawyerAgent("John Doe (Defendant)", DEFENDENT_SYSTEM)
plaintiff = LawyerAgent("TechCorp (Plaintiff)", PLAINTIFF_SYSTEM)
judge = LawyerAgent("Judge Williams", JUDGE_SYSTEM)

#accesing the data.csv
import pandas as pd

# Load dataset
cases_df = pd.read_csv("data.csv")

for index,row in cases_df.iterrows():
    case_background = row['text']  # or whatever column your case text is in
print(case_background)
# Case background
case_background = (
 "CASE THAT I WILL TAKE FROM THE DATASET"
)

def print_header(title):
    print("\n" + "="*50)
    print(f"== {title.upper()} ==")
    print("="*50 + "\n")

def simulate_trial():
    print_header("Court Session Begins")
    opening = judge.respond(
        "The court is now in session for State v. John Doe. "
        "This is a case regarding alleged theft of trade secrets. "
        "Prosecution, you may begin with your opening statement."
    )
    judge.print_colored(f"JUDGE: {opening}", 34)  # Blue for judge

    pros_open = prosecution_lawyer.respond(
        f"Your Honor, members of the court, {case_background} "
        "We will prove beyond reasonable doubt that the defendant "
        "stole valuable intellectual property from TechCorp."
    )
    prosecution_lawyer.print_colored(f"PROSECUTION: {pros_open}", 31)  # Red for prosecution
    

    defense_open = defense_lawyer.respond(
        "Your Honor, the defense maintains Mr. Doe's complete innocence. "
        "The prosecution's case rests entirely on circumstantial evidence. "
        "We will show there are legitimate explanations for the data transfer "
        "and no proof any trade secrets were taken."
    )
    defense_lawyer.print_colored(f"DEFENSE: {defense_open}", 32)  # Green for defense
    

    print_header("Prosecution Direct Examination")
    judge_instruction = judge.respond(
        "Prosecution, you may call your first witness."
    )

    judge.print_colored(f"JUDGE: {judge_instruction}", 34)

    plaintiff_testimony = plaintiff.respond(
        "As CTO of TechCorp, I can confirm John Doe had access to our "
        "proprietary algorithms. The timing of his massive data download "
        "just before resigning is highly suspicious."
    )
    plaintiff.print_colored(f"PLAINTIFF: {plaintiff_testimony}", 35)  # Purple

    print_header("Defense Cross-Examination")
    defense_q1 = defense_lawyer.respond(
        "Isn't it true that employees regularly download large datasets "
        "for legitimate work purposes?"
    )
    defense_lawyer.print_colored(f"DEFENSE: {defense_q1}", 32)
    opinions=[]
    for juror in jurry_personas:
        response=juror.respond(
            defense_q1
        )
        opinions.append((juror.name, response)) 
    for juror in jurry_personas:
        combined= "\n".join([f"{name}: {op}" for name, op in opinions])
        reflection_prompt = f"As {juror.name}, here's what others said:\n{combined}\nNow, what's your final opinion and vote?"
        final_response=juror.respond(reflection_prompt)
        juror.print_colored(f"{juror.name}: {final_response}", 36)  
    #adding the response of each juror to the list of opinions
    plaintiff_response = plaintiff.respond(defense_q1)
    plaintiff.print_colored(f"PLAINTIFF: {plaintiff_response}", 35)

    pros_redirect = prosecution_lawyer.respond(
        "But isn't 15GB an unusually large amount, especially just before resignation?"
    )
    prosecution_lawyer.print_colored(f"PROSECUTION: {pros_redirect}", 31)
    
    plaintiff_final = plaintiff.respond(pros_redirect)
    plaintiff.print_colored(f"PLAINTIFF: {plaintiff_final}", 35)

    print_header("Defense Case")
    judge_instruction = judge.respond(
        "Defense, you may call your first witness."
    )
    judge.print_colored(f"JUDGE: {judge_instruction}", 34)

    defendant_testimony = defendant.respond(
        "I downloaded those files to prepare my quarterly reports. "
        "They contained raw data, not algorithms. I never shared "
        "anything with InnovateCo."
    )
    defendant.print_colored(f"DEFENDANT: {defendant_testimony}", 33)  # Yellow

    print_header("Prosecution Cross-Examination")
    pros_q1 = prosecution_lawyer.respond(
        "If this was routine, why didn't you mention it during your exit interview?"
    )
    prosecution_lawyer.print_colored(f"PROSECUTION: {pros_q1}", 31)
    
    defendant_response = defendant.respond(pros_q1)
    defendant.print_colored(f"DEFENDANT: {defendant_response}", 33)

    print_header("Closing Arguments")
    pros_closing = prosecution_lawyer.respond(
        "In summary, the evidence shows the defendant had means, opportunity, "
        "and motive to steal trade secrets. The timing is damning."
    )
    prosecution_lawyer.print_colored(f"PROSECUTION: {pros_closing}", 31)
    
    defense_closing = defense_lawyer.respond(
        "The prosecution has failed to meet its burden of proof. "
        "There's reasonable doubt about what was downloaded and why."
    )
    defense_lawyer.print_colored(f"DEFENSE: {defense_closing}", 32)
    
    # 10. Judge's instructions
    # verdict = judge.respond(
    #     "Having heard all evidence, I find the prosecution has not proven "
    #     "beyond reasonable doubt that trade secrets were stolen. "
    #     "Case dismissed."
    # 
    verdict=judge.respond(
        "The jury is instructed to deliberate on the evidence presented. "
        "Remember, the burden of proof lies with the prosecution."
    )
    verdict=judge.respond(
        "We have enough evidence to prove that the defendant is guilty. "
        "The defendant is found guilty of theft of trade secrets."
        "The defendant is sentenced to 5 years in prison."
    )
    judge.print_colored(f"JUDGE: {verdict}", 34)

if __name__ == "__main__":
    simulate_trial()