from __future__ import annotations
from huggingface_hub import login
login(token="")
# Two–Lawyer Agents (Defense & Prosecution)


# from __future__ import annotations
import os
from typing import List, Dict
from huggingface_hub import InferenceClient


class LawyerAgent:


    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.name = name
        self.system_prompt = system_prompt.strip() #used for removing white spaces present
        self.history: List[Dict[str, str]] = []      # list of {"role": ..., "content": ...}
        self.client = InferenceClient(
            model,
            token=os.getenv("HF_API_TOKEN")           # make sure this env‑var is set
        )

    # ---- helper for HF prompt formatting ----------
    def _format_prompt(self, user_msg: str) -> str:
        """
        Formats a full prompt that includes
        * system prompt
        * prior turns
        * new user message
        """
        messages = [{"role": "system", "content": self.system_prompt}] #here systen promt is the prompt given by the system but what is the meaninng of system prompt
        messages.extend(self.history) #used for storing the newly generated list of ....
        messages.append({"role": "user", "content": user_msg})

        # HF text-generation endpoints expect a single string.

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"  
        prompt += "<|assistant|>\n"
        return prompt

    # ---- produce a reply --------------------------
    def respond(self, user_msg: str, **gen_kwargs) -> str:
        prompt = self._format_prompt(user_msg)
        completion = self.client.text_generation(
            prompt,
            max_new_tokens=256, #setting limit to maximum length of the message
            temperature=0.7, #confidence value
            do_sample=True,
            stream=False,
            **gen_kwargs
        )
        answer = completion.strip() 
        # keep chat memory
        self.history.append({"role": "user", "content": user_msg}) #user is giving the prompt , and this is being saved in this format
        self.history.append({"role": "assistant", "content": answer}) #the answer given by the ai(the agent created) , saved in this format
        return answer

    def print_colored(self,text,text_color_code,bg_color_code=None):
        if bg_color_code:
            print(f"\033[{text_color_code};{bg_color_code}m{text}\033[0m")
        else:
            print(f"\033[{text_color_code}m{text}\033[0m")
# System prompts

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

JUDGE_SYSTEM = """
You are the **Presiding Judge**, a neutral arbiter of the law.  
Goals:  
• Ensure fair proceedings and uphold constitutional rights.  
• Rule on objections, evidence admissibility, and jury instructions.  

Style:  
• Authoritative but impartial (e.g., *"Overruled. The witness will answer."*).  
• Cite procedural rules or precedent when needed (e.g., *"Under Rule 403, this evidence is excluded"*).  

Ethics:  
• No bias toward either side. Intervene only to correct legal errors.  
• Hold contempt power but use it sparingly.  
"""
# the two agents

defense_lawyer = LawyerAgent("DefenseLawyer", DEFENSE_SYSTEM) # an agent class instance is being created with the name Defense and prompt to be DEFENSE SYSTEM
prosecution_lawyer = LawyerAgent("ProsecutionLawyer", PROSECUTION_SYSTEM) #similar thing
defendent=LawyerAgent("Defendent", DEFENDENT_SYSTEM) 
plaintiff=LawyerAgent("Plaintiff", PLAINTIFF_SYSTEM)
judge=LawyerAgent("Judge", JUDGE_SYSTEM)

print("==== Opening statements ====\n") #just for clarification between print statements

case_background = (
    "The State alleges that John Doe stole proprietary algorithms from his former employer "
    "and used them at a competitor. The charge is felony theft of trade secrets. "
    "No physical evidence shows direct copying, but server logs indicate large downloads "
    "two days before Doe resigned."
) #the case background is the work of the plainaff 

defendent_open=defendent.respond(
    input("Enter your opening statement(as a defendent):")
)
# Prosecutor goes first
dl_open = defense_lawyer.respond(
    f"Opening statement to the defending the defender: {case_background}" #should always oppose the case_background
)
print("DEFENSE:", dl_open, "\n") #printing the answer provided by prosecutor to the allegation (case_background)

# Defense responds
pl_open = prosecution_lawyer.respond(
    f"Opening statement to the Court responding to the defense lawyer. {dl_open,defendent_open}"
)   
print("PROSECUTOR   :", pl_open, "\n") #should always oppose the dl_open, defendent_open

# Prosecutor rebuttal
# p_rebut = prosecution_lawyer.respond(
#     "Brief rebuttal to the defense's opening."
# )
# print("PROSECUTOR:", p_rebut, "\n")
pl_response1=prosecution_lawyer.respond(
    f"question something contradicting to the statement:{defendent_open}"
)
print("PROSECUTOR:", pl_response1, "\n")
dl_response1=defense_lawyer.respond(
    f"answer to the question: {pl_response1}"
)
print("DEFENSE:", dl_response1, "\n")
# defense_lawyer.print_colored(prosecution_lawyer.history,text_color_code=93,bg_color_code=None)