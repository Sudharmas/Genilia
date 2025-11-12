import uvicorn
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')

BASE_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"

ADAPTER_PATH = os.path.join(root_dir, "genilia-qwen2-expert")

print("--- Initializing Fine-Tuned SLM Agent ---")

print(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

print(f"Loading adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)


print("Merging adapter into base model...")
model = model.merge_and_unload()
print("Model merge complete.")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Creating text generation pipeline...")

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("--- Fine-Tuned Agent is Ready ---")

app = FastAPI(
    title="Genilia Fine-Tuned Agent",
    description="Microservice for serving the custom-trained Qwen2 SLM.",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def get_status():
    return {"status": "ok", "message": "Fine-Tuned Agent is running!"}


@app.post("/query")
def query_finetuned_model(request: QueryRequest):
    """
    Runs a query against the custom fine-tuned SLM.
    """
    print(f"\nReceived query for SLM: {request.question}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided data."},
        {"role": "user", "content": request.question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        full_text = outputs[0]['generated_text']
        answer = full_text.split("<|im_end|>\n<|im_start|>assistant\n")[1]

        print(f"Generated answer: {answer}")
        return {"question": request.question, "answer": answer}

    except Exception as e:
        print(f"Error during model inference: {e}")
        return {"error": f"An error occurred: {e}"}, 500


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)