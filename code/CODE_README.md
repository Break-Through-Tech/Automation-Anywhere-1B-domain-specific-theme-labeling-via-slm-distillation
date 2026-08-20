# Domain-Specific Theme Labeling via SLM Distillation

**Break Through Tech AI Studio — Fall 2026**  
**Company:** Automation Anywhere × Aisera  
**Advisor:** Staff Data Scientist, Automation Anywhere

---

## What This Project Does

Large frontier LLMs (like GPT-4 or Claude) can generate high-quality descriptive labels for clusters of support tickets — but they are slow and expensive at production scale.

This project **distills** that capability into a smaller open-source language model (SLM) using a technique called **LLM-supervised fine-tuning** (also called knowledge distillation in the broad sense):

1. A frontier LLM labels clusters of IT/HR/CX support tickets
2. A small open-source SLM is fine-tuned on those labels using LoRA/QLoRA
3. We measure whether the fine-tuned SLM can replicate near-teacher label quality at a fraction of the cost and latency

---

## Project Phases

| Phase | Focus | Duration | Dataset |
|-------|-------|----------|---------|
| **Phase 1** | Pipeline setup and verification | Weeks 1–2 | Bitext public dataset (~500 IT tickets) |
| **Phase 2** | Real experiments and analysis | Weeks 3–10 | Enterprise dataset (provided by advisor) |
| **Phase 3** | Results, story, research paper | Weeks 11–12 | — |

**You are here: Phase 1.** The goal is to get the full pipeline running end-to-end and understand every step — not to produce publishable results.

---

## Models

### Phase 1 Student SLMs

| Model | Parameters | Device mode | HuggingFace ID |
|-------|-----------|-------------|----------------|
| SmolLM2-360M | 0.36B | `local_cpu` (smoke test) | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| SmolLM2-1.7B | 1.7B | `local_mps` (Mac) | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| **Phi-3.5-Mini** | **3.8B** | **`colab` (primary)** | `microsoft/Phi-3.5-mini-instruct` |

### Teacher / Judge LLMs

| Role | Model | Provider |
|------|-------|----------|
| Teacher (generates labels) | `claude-haiku-4-5` | Anthropic |
| Evaluation judge | `claude-haiku-4-5` | Anthropic (cross-model) |

---

## Repository Structure

```
.
├── README.md                        ← you are here
├── main.py                          ← entry point: python main.py --phase 1
├── requirements.txt                 ← base dependencies (M1 Mac + Colab)
├── requirements_colab.txt           ← Colab GPU additions (QLoRA, Unsloth)
│
├── configs/
│   ├── phase1_config.yaml           ← all Phase 1 settings (edit this)
│   └── phase2_config.yaml           ← Phase 2 skeleton
│
├── notebooks/
│   └── 00_session_setup.ipynb       ← run this at the start of every Colab session
│
├── phase1/                          ← Phase 1 code (complete, read and run)
│   ├── pipeline.py                  ← orchestrates all steps
│   ├── data/
│   │   ├── schema.py                ← column name constants (single source of truth)
│   │   ├── clustering.py            ← download Bitext, embed, UMAP, HDBSCAN
│   │   └── preprocessing.py         ← group by cluster, select top-k
│   ├── prompts/
│   │   └── templates.py             ← P1–P5 prompt templates
│   ├── labeling/
│   │   └── frontier_llm.py          ← API calls with rate limiting
│   ├── finetuning/
│   │   ├── dataset.py               ← labeled CSV → training JSONL
│   │   └── trainer.py               ← hardware-aware LoRA/QLoRA trainer
│   └── evaluation/
│       ├── metrics.py               ← cosine similarity, BERTScore, ROUGE-L
│       ├── llm_judge.py             ← LLM-as-judge evaluation
│       └── business_eval.py         ← latency, cost, throughput
│
├── phase2/                          ← Phase 2 (students write code here)
│   └── README.md
│
└── data/                            ← NOT in GitHub — lives on Google Drive
    ├── raw/                         ← downloaded Bitext data
    ├── processed/                   ← grouped cluster CSV, labeled CSV
    └── checkpoints/                 ← embeddings, UMAP projections (pkl files)
```

> **Note:** The `data/` and `outputs/` directories are in `.gitignore`. They live on your Google Drive, not in GitHub. Only code, configs, and notebooks are committed.

---

## Setup: Local (M1/M5 Mac)

Follow these steps exactly in order.

### Prerequisites

1. **Python 3.10 or 3.11** — check with `python3 --version`. If not installed, download from [python.org](https://www.python.org/downloads/).
2. **Git** — check with `git --version`. Install via Xcode tools: `xcode-select --install`
3. **VS Code** (recommended editor) — download from [code.visualstudio.com](https://code.visualstudio.com)

### Step 1: Clone the repository

```bash
# Open Terminal, navigate to where you want the project
cd ~/Documents

# Clone the repo (replace with the actual BTT repo URL)
git clone https://github.com/Break-Through-Tech/Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation.git

# Enter the directory
cd Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation
```

### Step 2: Create a virtual environment

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate it (you must do this every time you open a new terminal)
source .venv/bin/activate

# Confirm it's active — you should see (.venv) in your prompt
```

### Step 3: Install dependencies

```bash
# Install base requirements
pip install --upgrade pip
pip install -r requirements.txt
```

> **M1/M5 known issue — umap-learn:** If `umap-learn` installation fails with a `llvm` error, run:
> ```bash
> brew install llvm
> pip install umap-learn
> ```
> If you don't have Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

> **M1/M5 known issue — hdbscan:** If hdbscan fails to compile, try:
> ```bash
> pip install hdbscan --no-build-isolation
> ```

### Step 4: Set up environment variables

Create a `.env` file in the project root (this file is in `.gitignore` — never commit it):

```bash
# Create the .env file
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
EOF
```

Replace the placeholder values with your actual keys:
- **Anthropic API key:** [console.anthropic.com](https://console.anthropic.com) → API Keys
- **OpenAI API key:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **HuggingFace token:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (read access is enough)

### Step 5: Configure for local mode

Open `configs/phase1_config.yaml` and set:

```yaml
device_mode: "local_mps"   # for M1/M5 Mac with MPS
```

The default model (`microsoft/Phi-3.5-mini-instruct`, 3.8B) fits in 16GB unified memory on M1 with MPS. For a quick smoke test (CPU only, 360M model):

```yaml
device_mode: "local_cpu"
student_slm:
  model_id: "HuggingFaceTB/SmolLM2-360M-Instruct"
```

### Step 6: Run a smoke test

```bash
# Activate venv if not already active
source .venv/bin/activate

# Run Phase 1 pipeline
python main.py --phase 1 --config configs/phase1_config.yaml
```

You should see step-by-step progress printed to the terminal. The first run downloads the Bitext dataset and the SLM — this takes a few minutes depending on your internet speed.

---

## Setup: Google Colab

Follow these steps at the start of **every new Colab session**. The `00_session_setup.ipynb` notebook in the `notebooks/` folder contains all of these cells pre-written — open it first.

### Before your first session (do once)

**a) Create a HuggingFace account and token**
1. Go to [huggingface.co](https://huggingface.co) and sign up with your Cornell email
2. Go to Settings → Access Tokens → New Token → name it "colab" → Read access → Create
3. Copy the token — you'll need it in Step c below

**b) Get your API keys**
- **Anthropic (Claude):** [console.anthropic.com](https://console.anthropic.com) → sign up → API Keys → Create Key
- **OpenAI (optional):** [platform.openai.com](https://platform.openai.com) → API Keys → Create Key

**c) Set up Colab Secrets**
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Click the **key icon** (🔑) in the left sidebar
3. Add these secrets (click "+ Add new secret" for each):

| Secret name | Value |
|-------------|-------|
| `ANTHROPIC_API_KEY` | your Claude API key |
| `OPENAI_API_KEY` | your OpenAI key (if you have one) |
| `HF_TOKEN` | your HuggingFace token |
| `GITHUB_PAT` | your GitHub Personal Access Token (see below) |

**d) Create a GitHub Personal Access Token (PAT)**
1. Go to [github.com](https://github.com) → click your profile photo → Settings
2. Scroll to the bottom of the left sidebar → Developer settings
3. Personal access tokens → Tokens (classic) → Generate new token
4. Set expiration: 90 days (covers the whole program)
5. Check the `repo` scope (full control of repositories)
6. Click Generate token — **copy it immediately, it's shown only once**
7. Save it as the `GITHUB_PAT` Colab secret (step c above)

**e) Set up your Google Drive folder**
1. Go to [drive.google.com](https://drive.google.com) using your Cornell email
2. Create a folder called `slm-distillation`
3. Inside it, create these subfolders: `data/raw`, `data/processed`, `data/checkpoints`, `outputs/labels`, `outputs/models`, `outputs/evaluation`, `hf_cache`

**f) Create your student branch**

Open `00_session_setup.ipynb` in Colab and run the "First session only" cell:

```python
# Run this ONE TIME to create your branch
import subprocess, os
from google.colab import userdata

pat = userdata.get('GITHUB_PAT')
repo_url = f"https://{pat}@github.com/Break-Through-Tech/Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation.git"

os.makedirs('/content/project', exist_ok=True)
subprocess.run(['git', 'clone', repo_url, '/content/project'], check=True)
os.chdir('/content/project')

# ← CHANGE THIS to your pair's branch name (e.g., "pair-1/smollm2-1.7b")
BRANCH_NAME = "pair-X/model-name"

subprocess.run(['git', 'checkout', '-b', BRANCH_NAME], check=True)
subprocess.run(['git', 'push', '-u', 'origin', BRANCH_NAME], check=True)
print(f"Branch '{BRANCH_NAME}' created and pushed to GitHub.")
```

### Every subsequent session (run 00_session_setup.ipynb top to bottom)

The setup notebook handles everything. It takes **5–8 minutes** total:

| Step | Time | What happens |
|------|------|-------------|
| Install packages | ~3 min | `pip install` from requirements files |
| Mount Google Drive | ~15 sec | One browser click to authorise |
| Set HF cache to Drive | instant | `HF_HOME` set to Drive path |
| Load API keys from Secrets | instant | Reads from Colab Secrets panel |
| Pull latest code from GitHub | ~15 sec | `git pull` from your branch |
| Verify GPU | instant | `nvidia-smi` — confirm T4 and 16GB VRAM |

> **Tip:** If Colab gives you a K80 GPU instead of T4, disconnect (Runtime → Disconnect) and reconnect. T4 is much faster and you need 16GB for 3.8B models.

### Switching to Colab mode

In `configs/phase1_config.yaml`, update:

```yaml
device_mode: "colab"

paths:
  drive_root: "/content/drive/MyDrive/slm-distillation"
```

Everything else — model downloads, checkpoints, outputs — automatically routes to your Drive.

---

## Running the Pipeline

```bash
python main.py --phase 1 --config configs/phase1_config.yaml
```

To skip steps you've already completed (e.g., re-run only evaluation):

```yaml
# In phase1_config.yaml:
pipeline:
  run_clustering:       false   # already done, checkpoint exists
  run_preprocessing:    false   # already done
  run_label_generation: false   # already done
  run_finetuning:       false   # already done
  run_baseline_eval:    true    # re-run this
  run_finetuned_eval:   true    # re-run this
  run_llm_judge:        true
  run_business_eval:    true
```

---

## Working with GitHub (daily workflow)

### Saving your changes

```bash
# In Colab terminal or local terminal:
git add .
git commit -m "Brief description of what you changed"
git push
```

### Getting the advisor's latest updates (from main branch)

```bash
git fetch origin
git merge origin/main
```

> Do this at the start of each week — the advisor may have pushed bug fixes or new features.

### Sharing results with your pair partner

Both pair members work on the **same branch**. If your partner pushed changes:

```bash
git pull origin your-branch-name
```

---

## Output Files

All outputs save to `{drive_root}/outputs/`:

| File | Contents |
|------|----------|
| `labels/bitext_clustered.csv` | Clustered tickets with cluster IDs |
| `labels/bitext_labeled.csv` | Clusters with all 5 teacher-generated labels |
| `evaluation/baseline_predictions.jsonl` | Pre-distillation SLM predictions |
| `evaluation/finetuned_predictions.jsonl` | Post-distillation SLM predictions |
| `evaluation/metrics_summary.csv` | All evaluation metrics in one table |
| `evaluation/business_eval.csv` | Latency, cost, throughput |
| `models/lora_adapter/` | Fine-tuned LoRA adapter weights |

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Cluster** | A group of support tickets sharing a common theme |
| **Cluster label** | A 5–15 word description of a cluster's theme |
| **Teacher LLM** | Frontier model (Claude Haiku) that generates gold labels |
| **Student SLM** | Small open-source model being fine-tuned |
| **Distillation** | Training the SLM to replicate the teacher's labeling |
| **LoRA** | Efficient fine-tuning: trains small adapters, freezes base model |
| **QLoRA** | LoRA + 4-bit quantisation (Colab only, saves VRAM) |
| **MPS** | Apple's GPU backend for PyTorch on M1/M2/M5 chips |
| **Top-k** | The k tickets closest to the cluster centroid |

---

## Troubleshooting

**"CUDA out of memory" in Colab**
- Reduce `per_device_train_batch_size` to 2 in the config
- Ensure `gradient_checkpointing: true`
- Try Runtime → Disconnect, reconnect, hope for T4 instead of K80

**"bitsandbytes" import error on Mac**
- Expected — bitsandbytes requires CUDA. Use `device_mode: "local_mps"` instead.

**umap-learn or hdbscan install fails on Mac**
- See Step 3 in the local setup guide above.

**API rate limit errors**
- Increase `sleep_between_calls` in the config. Start with 2.0 seconds.

**Colab session ends mid-training**
- Checkpoints save after each epoch. Restart the session, run `00_session_setup.ipynb`, set `run_finetuning: true` and `run_clustering: false` in the config. Training resumes from the last checkpoint.

---

## Contact

Questions? Reach out to your advisor or post in the project Slack channel.
