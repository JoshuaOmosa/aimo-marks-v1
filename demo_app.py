import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

# --- STEP 1: MEMORY SANITIZATION ---
def clear_vram():
    if 'trainer' in globals(): del trainer
    if 'model' in globals(): del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("VRAM Purged. System ready for demo.")

clear_vram()

# --- STEP 2: LOAD THE BRAIN (AIMO-MARKS-V1) ---
model_id = "deepseek-ai/deepseek-math-7b-rl"
adapter_path = "./marks_v1_model"  

print("Loading AIMO-Marks-V1 Architecture...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load base model + your trained adapters
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
demo_model = PeftModel.from_pretrained(base_model, adapter_path)

# --- STEP 3: OUTPUT CLEANING LOGIC ---
def format_reasoning(text):
    # Add spaces between camelCase words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Add space after punctuation if missing
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
    
    # Handle the "Infinite Loop" fix
    if "The answer is" in text:
        parts = text.split("The answer is")
        text = parts[0] + "The answer is" + parts[1].split(".")[0] + "."
    return text

# --- STEP 4: THE DEMO ENGINE ---
def aimo_demo_solve(problem):
    if not problem or not problem.strip():
        return "Please enter a valid math problem."
    
    # Formatting the prompt exactly as it was during training
    full_prompt = f"Problem: {problem}\n\nSolution: "
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output_tokens = demo_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # Extract only the AI's answer, not the prompt
    if "Solution:" in raw_response:
        clean_response = raw_response.split("Solution:")[-1].strip()
    else:
        clean_response = raw_response.strip()
    
    return format_reasoning(clean_response)

# --- STEP 5: GRADIO UI (THE FRONTEND) ---
demo = gr.Interface(
    fn=aimo_demo_solve,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Ask AIMO-Marks-V1 a math problem...",
        label="User Input"
    ),
    outputs=gr.Textbox(label="AIMO-Marks-V1 Reasoning Chain", lines=15),
    title="AIMO-Marks-V1 | Mathematical Reasoning Engine",
    description="Fine-tuned DeepSeek-7B for step-by-step mathematical reasoning.",
    theme="soft",
    examples=[
        ["A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?"],
        ["What is the sum of all prime numbers less than 20?"],
        ["If x + y = 10 and x * y = 21, find x² + y²."],
    ]
)

# Launch with a public link
if __name__ == "__main__":
    demo.launch(share=True)