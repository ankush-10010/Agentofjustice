Agent of Justice: A Courtroom Simulation using LLMs
Agent of Justice is a Large Language Model (LLM)-based courtroom simulation project. It creates an artificial yet interactive trial experience by simulating the roles of legal professionals such as Judge, Lawyers, Plaintiff, and Defendant using multiple LLM instances that talk to each other based on custom-designed prompts.

This repository evolves over four progressively advanced models, adding new features such as voice output, fine-tuning on real legal data, and jury deliberation.


Project Structure 
Cynaptics_Project/
    basic_code.py  #code provided
    data.csv       #provided dataset
    model1.py   
    model2.py
    model3.py
    model4.py
    README.md
    test1.py
    <!-- test_trained_model.py -->  #for testing if the trained model actually works
    test_voice.py                   #for testing the voice features
    training_model.py               #for fine_tuning the model
    phi3-casehold-judge/            
        runs/
            Apr18_18-42-12_WIN-G0AGT04LNRH/
                events.out.tfevents.1744981932.WIN-G0AGT04LNRH.39728.0
    saved_model/                    #trained model (fine tuned)
        finetunedphi3/
            adapter_config.json
            adapter_model.safetensors
            added_tokens.json
            README.md
            special_tokens_map.json
            tokenizer.json
            tokenizer.model
            tokenizer_config.json
    __pycache__/
        ANSI.cpython-312.pyc
🧠 Features by Model Version


Model 1: Basic Simulation
Instances of a base LLM (microsoft/Phi-3-mini-4k-instruct)
Manually structured trial flow
Agents (Judge, Plaintiff, Defendant, Lawyers) interact through prompts


Model 2: Enhanced Prompting + Voice Output
Improved prompt engineering for more realistic interactions
Adds text-to-speech for opening statements using pyttsx3 or gTTS
Retains Model 1 structure with additional clarity and role instructions



Model 3: Fine-tuned Judge Model
Fine-tuned on the LexGLUE - CaseHold dataset (legal reasoning)
Judge agent uses this fine-tuned model for verdicts and analysis
Training code: training_model.py
Shows better understanding of legal context in trials



Model 4: Custom Jury Deliberation
Adds multiple jury agents as LLM instances
Each juror discusses and evalutes the case
Jurors individually vote on the verdict
Not fine-tuned but runs using engineered prompt templates

other models from model 5-7 RE underdeveloped 

Model 8:
it has extra features such as:
API Management:
Quota tracking with warnings
Exponential backoff retries
Response caching (LRU cache)

Model 10:
Tried something new in this model by integrating groq api
Improved Legal Process Fidelity
Structured Adjudication:
Enforces 4-step judicial analysis (evidence → standards → precedent → verdict)
Mandates binary APPROVED/REJECTED outcome (clearer than v2's numeric 1/0)

Enhanced Evidence Handling:
Formal witness examination sequences
Direct/cross-examination protocols
Witness-specific background integration
Case Summarization:

Automatic summary generation between phases
Last 10 interactions tracked (previously full history)


but didn't work as expected due to no quota management


Input Processing:
Case text parsing
Entity extraction (parties, charges, evidence)
Legal context matching


Verdict Validation:
Checks for required sections
Consistency between evidence strength and verdict


Automated Reporting:
Session timestamping
Statistics (guilty/not-guilty ratios)
Validation issue flagging


Tech Stack
Python
Hugging Face Transformers (microsoft/Phi-3-mini-4k-instruct)
torch
pyttsx3 / gTTS (for voice synthesis)


Dataset Used for Fine-Tuning
LexGLUE / CaseHold
Source: Hugging Face
Contains legal case holdings and facts for training on legal judgments.



Contributors
Ankush Raj 

