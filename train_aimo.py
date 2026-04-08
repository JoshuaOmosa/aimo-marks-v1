import torch
import os
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

free_memory()

# ── 1. LOAD MODEL ────────────────────────────────────────────────────────────
model_id = "deepseek-ai/deepseek-math-7b-rl"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)


model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True,
    clean_up_tokenization_spaces=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

free_memory()
print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── 2. LORA SETUP ────────────────────────────────────────────────────────────
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
free_memory()

# ── 3. DATASET ────────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train[:500]")  

# Check dataset structure
print(f"Dataset columns: {dataset.column_names}")
print(f"Sample problem: {dataset[0]['problem'][:100]}")
print(f"Sample solution: {dataset[0]['solution'][:100]}")

PROMPT_TEMPLATE = """Problem: {problem}

Solution: {solution}

"""

MAX_LENGTH = 512

def tokenize_fn(batch):
    texts = []
    for p, s in zip(batch['problem'], batch['solution']):
        # Clean the text
        p = p.strip()
        s = s.strip()
        
        # Format properly
        text = PROMPT_TEMPLATE.format(problem=p, solution=s)
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    return tokenized

print("Tokenizing dataset...")
tokenized_ds = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=20,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

print(f"Tokenized dataset size: {len(tokenized_ds)}")
print(f"Sample token length: {len(tokenized_ds[0]['input_ids'])}")

del dataset
free_memory()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ── 4. TRAINING ARGS (FIXED - removed invalid parameters) ────────────────────
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,  # Just 1 epoch for testing
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
    report_to="none",
    remove_unused_columns=False,
    prediction_loss_only=True,  
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    args=training_args,
)

# Memory report
free_memory()
if torch.cuda.is_available():
    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── 5. TEST BEFORE TRAINING ───────────────────────────────────────────────────
print("\n─── TESTING BASE MODEL BEFORE TRAINING ───")
model.eval()
test_prompt = "Problem: What is 2+2?\nSolution:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    test_out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
test_result = tokenizer.decode(test_out[0], skip_special_tokens=True)
print(f"Base model output: {test_result}")
print("-" * 50)

# ── 6. TRAIN ─────────────────────────────────────────────────────────────────
print("\n─── TRAINING STARTING ───")
try:
    trainer.train()
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    raise

# ── 7. TEST AFTER TRAINING ───────────────────────────────────────────────────
print("\n─── TESTING AFTER TRAINING ───")
model.eval()
test_prompt = "Problem: What is 2+2?\nSolution:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    test_out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
test_result = tokenizer.decode(test_out[0], skip_special_tokens=True)
print(f"Trained model output: {test_result}")
print("-" * 50)

# ── 8. TEST WITH ACTUAL MATH PROBLEM ──────────────────────────────────────────
print("\n─── TESTING WITH REAL PROBLEM ───")
test_prompt = """Problem: A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?

Solution:"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    test_out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
test_result = tokenizer.decode(test_out[0], skip_special_tokens=True)
# Remove the prompt from output
test_result = test_result.replace(test_prompt, "").strip()
print(f"Bat and ball problem solution:\n{test_result}")
print("-" * 50)

# ── 9. SAVE ──────────────────────────────────────────────────────────────────
save_path = "./marks_v1_model"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Saved to {save_path}")

# Verify save worked
adapter_file = os.path.join(save_path, "adapter_model.safetensors")
if os.path.exists(adapter_file):
    size = os.path.getsize(adapter_file) / 1e6
    print(f"✅ Adapter saved successfully! Size: {size:.2f} MB")
else:
    # Check for .bin format
    adapter_file = os.path.join(save_path, "adapter_model.bin")
    if os.path.exists(adapter_file):
        size = os.path.getsize(adapter_file) / 1e6
        print(f"✅ Adapter saved successfully! Size: {size:.2f} MB")
    else:
        print("❌ ERROR: Adapter not saved properly!")
        print(f"Contents of {save_path}: {os.listdir(save_path)}")

print("\n✅ Training pipeline complete!")