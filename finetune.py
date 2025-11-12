
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,

)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


DATASET_PATH = "finetune_dataset.jsonl"

BASE_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"

NEW_MODEL_NAME = "genilia-qwen2-expert"


print(f"Loading dataset from: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_chat_template(sample):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided data."},
        {"role": "user", "content": sample['question']},
        {"role": "assistant", "content": sample['answer']}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

print("Formatting dataset with Qwen2 chat template...")
formatted_dataset = dataset.map(format_chat_template)


print(f"Loading base model: {BASE_MODEL_ID}")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = False

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)

print("Applying PEFT/LoRA adapters to the model...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=training_arguments,
)

print("\n--- STARTING FINE-TUNING ---\n")
trainer.train()
print("\n--- FINE-TUNING COMPLETE ---")

print(f"Saving new model adapter to: {NEW_MODEL_NAME}")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("\n--- All Done! ---")
print(f"Your new model is saved in the '{NEW_MODEL_NAME}' folder.")