from datasets import load_dataset

dataset = load_dataset("coastalcph/lex_glue", "case_hold")

# print(dataset["train"][12])

def format_casehold(example):
    question = example["context"]
    options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(example["endings"])])
    correct = chr(65 + example["label"])
    
    return {
        "prompt": f"<|user|>\nGiven the case:\n{question}\n\nWhat is the most legally sound conclusion?\n{options}\n\n<|assistant|>",
        "response": f"The correct conclusion is option {correct}."
    }

formatted_dataset = dataset["train"].map(format_casehold)

small_formatted_dataset= formatted_dataset.shuffle(seed=42).select(range(100)) #ensures the reusability of the code , seed=42

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

model_name = "microsoft/phi-3-mini-4k-instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
model=AutoModelForCausalLM.from_pretrained(model_name,device_map="cpu")
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Tokenize
def tokenize(example):
    text = example["prompt"] + example["response"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = small_formatted_dataset.map(tokenize)

# LORA config
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["o_proj","qkv_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Training args
args = TrainingArguments(
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    output_dir="./phi3-casehold-judge"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,

    train_dataset=tokenized,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()


# After training is complete
output_dir = r"C:\CODE\Machine_Learning\Cynaptics_Project\test.py\fine_tuned_model_cpu"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
