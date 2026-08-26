# Local Setup Guide

> **Note:** Fine-tuning runs on Google Colab, not locally. Local setup is for
> editing code, running data processing, and quick smoke tests.

---

## Prerequisites

- Python 3.10 or 3.11 — [python.org](https://www.python.org/downloads/)
- Git — `xcode-select --install` (Mac)

---

## Setup steps

### 1. Clone the repository

```bash
git clone https://github.com/Break-Through-Tech/Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation.git
cd Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation
cd code
```

> All commands below run from inside `code/`. Open a new terminal? `cd` back here first.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate     # you'll see (.venv) in your prompt
```

Run this activation command every time you open a new terminal.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Do NOT install `requirements_colab.txt` locally** — it contains CUDA-only packages that don't work on Mac.

**If install fails on M1/M5 Mac:**

```bash
# umap-learn error:
brew install llvm && pip install umap-learn

# hdbscan error:
pip install hdbscan --no-build-isolation
```

### 4. Set up API keys

Create a `.env` file inside `code/`:

```bash
cat > .env << 'ENVEOF'
ANTHROPIC_API_KEY=your_claude_key_here
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_key_here
ENVEOF
```

- **Anthropic key:** [console.anthropic.com](https://console.anthropic.com) → API Keys → Create Key
- **HuggingFace token:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (Read)
- **OpenAI key:** optional

### 5. Verify the repo is complete

```bash
python verify_repo.py
```

If any files are missing, the output tells you exactly which ones.

### 6. Configure for local mode

In `configs/phase1_config.yaml`:

```yaml
device_mode: "local_mps"    # M1/M5 Mac
# device_mode: "local_cpu"  # any machine, smoke test only (very slow)
```

### 7. Run

```bash
python main.py --phase 1 --config configs/phase1_config.yaml --device_mode local_mps
```

For a quick smoke test (fast, uses tiny model on CPU):

```bash
python main.py --phase 1 --config configs/phase1_config.yaml \
  --device_mode local_cpu \
  --model "HuggingFaceTB/SmolLM2-360M-Instruct"
```

---

## Running the demo

After a training run completes:

```bash
python main.py --phase 1 --config configs/phase1_config.yaml \
  --device_mode local_mps \
  --mode demo \
  --adapter_dir outputs/YOUR_RUN_ID/models/lora_adapter
```

Type a ticket file path when prompted (e.g. `demo_tickets/it_password_reset.txt`).

---

## Skipping completed steps

If clustering and labeling are already done, set them to `false` in the config:

```yaml
pipeline:
  run_clustering:       false
  run_preprocessing:    false
  run_label_generation: false
  run_finetuning:       true
  run_baseline_eval:    true
  run_finetuned_eval:   true
  run_llm_judge:        true
  run_business_eval:    true
```

---

## Common issues

| Problem | Fix |
|---------|-----|
| `No module named 'phase1'` | You're not inside `code/`. Run `cd code` first. |
| `(.venv)` not in prompt | Virtual env not active. Run `source .venv/bin/activate`. |
| `ANTHROPIC_API_KEY not set` | `.env` file missing or not in `code/`. |
| Model download slow (first run) | Normal — models cache after first download. |
| BERTScore shows "skipped" | Expected locally. Enable it on Colab where it works correctly. |
