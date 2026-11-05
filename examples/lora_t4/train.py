"""Fine-tune a small LoRA adapter on a free Colab T4 and survive runtime loss.

Run it from Hermes:  "train examples/lora_t4/train.py on a Colab T4"

Everything it needs to resume goes to $HERMES_COLAB_OUTPUT_DIR, which hermes-colab syncs
home while the job runs and restores on a fresh runtime (HERMES_COLAB_RESUME=1).
Progress lines ("HERMES_PROGRESS {json}") show up in colab_status and the Desktop.
"""

import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", "150"))
OUT = os.environ.get("HERMES_COLAB_OUTPUT_DIR", "outputs")
CKPT = os.path.join(OUT, "checkpoint")
STATE = os.path.join(OUT, "state.json")


def progress(**data):
    print("HERMES_PROGRESS " + json.dumps(data), flush=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device} gpu={torch.cuda.get_device_name(0) if device == 'cuda' else '-'}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32)

state = {"epoch": 0, "losses": []}
if os.environ.get("HERMES_COLAB_RESUME") == "1" and os.path.exists(STATE) and os.path.isdir(CKPT):
    state = json.load(open(STATE))
    model = PeftModel.from_pretrained(base, CKPT, is_trainable=True)
    print(f"Resumed from checkpoint after epoch {state['epoch']}", flush=True)
else:
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.to(device).train()

data = load_dataset("yahma/alpaca-cleaned", split="train[:4000]")
texts = [f"### Instruction:\n{r['instruction']}\n{r['input']}\n### Response:\n{r['output']}" for r in data]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
total = EPOCHS * STEPS_PER_EPOCH

for epoch in range(state["epoch"], EPOCHS):
    started = time.time()
    for step in range(STEPS_PER_EPOCH):
        i = (epoch * STEPS_PER_EPOCH + step) * 8 % len(texts)
        batch = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = epoch * STEPS_PER_EPOCH + step + 1
        if step % 10 == 0:
            progress(step=done, total=total, fraction=round(done / total, 3),
                     label=f"epoch {epoch + 1}/{EPOCHS} · loss {loss.item():.3f}")
    model.save_pretrained(CKPT)  # adapter only: a few MB, synced home in seconds
    state = {"epoch": epoch + 1, "losses": state["losses"] + [round(loss.item(), 4)]}
    tmp = STATE + ".tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, STATE)
    print(f"epoch {epoch + 1} done in {time.time() - started:.0f}s, loss {loss.item():.3f}", flush=True)

model.eval()
prompt = "### Instruction:\nGive three tips for staying focused.\n\n### Response:\n"
ids = tok(prompt, return_tensors="pt").to(device)
sample = tok.decode(model.generate(**ids, max_new_tokens=80)[0], skip_special_tokens=True)
open(os.path.join(OUT, "sample.txt"), "w").write(sample)
print(sample, flush=True)
