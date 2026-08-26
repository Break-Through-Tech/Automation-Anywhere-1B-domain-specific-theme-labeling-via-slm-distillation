# Google Colab Setup Guide

Google Colab gives you a free cloud GPU (NVIDIA T4, 16 GB) in your browser.
Your code lives on GitHub, your outputs live on Google Drive, and Colab is just
the machine you use to run training. When a session ends, Drive keeps everything safe.

---
NOTE: Keep/Save all your keys in one notepad for easy access. 
## What you need (one-time, ~30 minutes)

### 1. Google account

Use your **Cornell email** (`netid@cornell.edu`) for all accounts below.

### 2. Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com) and sign up
2. Click **API Keys** → **+ Create Key**, name it anything
3. **Copy the key immediately** — it is shown only once (starts with `sk-ant-`)

### 3. HuggingFace token

1. Go to [huggingface.co](https://huggingface.co) and sign up
2. Profile → Settings → **Access Tokens** → **New token**
3. Type: **Read**, name it anything. Copy the token (starts with `hf_`).

### 4. GitHub Personal Access Token (PAT)

1. Go to [github.com](https://github.com) → your profile photo → **Settings**
2. Scroll the left sidebar to the bottom → **Developer settings**
3. **Personal access tokens → Tokens (classic) → Generate new token (classic)**
4. Expiration: **90 days**. Under Scopes, check **repo**.
5. Click **Generate token** — copy it immediately (starts with `ghp_`)

### 5. Colab Secrets

Colab Secrets store your keys securely so they are never in any file.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Press **Escape** to close any popup
3. In the left sidebar, click the **🔑 key icon** (Secrets)
4. For each row below: click **+ Add new secret**, fill in name and value,
   then **toggle "Notebook access" to ON (blue)**

> ⚠️ The toggle is OFF by default. If you forget to turn it ON,
> the code cannot read the key even though you added it.

| Secret name — type exactly | Value |
|---------------------------|-------|
| `ANTHROPIC_API_KEY` | Your Claude key (sk-ant-...) |
| `HF_TOKEN` | Your HuggingFace token (hf_...) |
| `GITHUB_PAT` | Your GitHub PAT (ghp_...) |
| `OPENAI_API_KEY` | OpenAI key (optional — skip if you don't have one) |

### 6. Create your branch on GitHub

1. Go to your repository on github.com
2. Click the branch dropdown button that shows **`main`**
3. Type your branch name in the text box (e.g. `pair-1/smollm2-1.7b`)
4. Click **"Create branch: pair-1/smollm2-1.7b from main"**

Your advisor will tell you which pair number and model name to use.

### 7. Set up Google Drive

1. Go to [drive.google.com](https://drive.google.com) with your Cornell email
2. Create a folder called `slm-distillation`

The setup notebook creates all subfolders automatically on first run.

---

## Every session — running the setup notebook

Run this notebook at the start of **every** Colab session. Takes 5–8 minutes.

### Open the notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Press **Escape** to close any dialog
3. In the **menu bar at the very top** of the page (the words
   `File  Edit  View  Insert  Runtime  Tools  Help`), click **File**
4. Click **Open notebook**
5. In the dialog, click the **GitHub** tab
6. First time: click **"Sign in with GitHub"** and authorise (one-time step)
7. Put the GitHub URL of the Repo in the Search Box
8. Search for the repository, select **your branch** from the dropdown,
   open `notebooks/00_session_setup.ipynb`

> **Shortcut from session 2 onwards:** File → Open notebook → **Recent**

### Run the cells in order

Click ▶ on each cell. Wait for it to finish before running the next.

| Cell | What it does | Time |
|------|-------------|------|
| **Cell 1** | Verify GPU — must show T4 (≥15 GB). If it shows K80: Runtime → Disconnect and delete runtime → reconnect. | instant |
| **Cell 2** | Mount Google Drive — a popup asks for permissions. **Click Allow on everything.** | 15 sec |
| **Cell 3** | Clone or pull repo — **change `BRANCH_NAME` to your branch before running** | 15 sec |
| **Cell 4** | Install packages — takes 3–4 minutes, normal to see progress bars | 3–4 min |
| **Cell 5** | Create Drive folders and set model cache | instant |
| **Cell 6** | Load API keys — all lines should show ✅ | instant |
| **Cell 7** | Verify setup — all checks should show ✅ | instant |

> **Cell 3 before Cell 4:** The install step reads `requirements.txt` from the
> cloned repo. On your first session the repo doesn't exist yet, so clone (Cell 3)
> before installing (Cell 4). The notebook labels remind you of this.

> **Google Drive permissions:** The popup lists several permissions including
> "See, edit, create, and delete all of your Google Drive files." This is Google's
> standard consent screen for Drive access — click **Allow** on everything.

When Cell 7 shows all ✅, scroll down to run the pipeline.

---

## Running the pipeline

### Full training run

```python
import os
CODE = os.environ.get('SLM_CODE_DIR', '/content/project/code')
!python "$CODE/main.py" \
    --phase 1 \
    --config "$CODE/configs/phase1_config.yaml" \
    --device_mode colab
```

Expected time on T4 (Phi-3.5-Mini, 5 epochs): **~45–55 minutes total**

### Evaluation only (model already trained)

Set `existing_run_dir` in the config to your run's folder path, set all
`run_clustering / run_preprocessing / run_label_generation / run_finetuning`
flags to `false`, and run the same command.

### Live demo

```python
import os
CODE    = os.environ.get('SLM_CODE_DIR', '/content/project/code')
DRIVE   = '/content/drive/MyDrive/slm-distillation'
ADAPTER = f'{DRIVE}/outputs/YOUR_RUN_ID/models/lora_adapter'  # ← update this

!python "$CODE/main.py" \
    --phase 1 \
    --config "$CODE/configs/phase1_config.yaml" \
    --device_mode colab \
    --mode demo \
    --adapter_dir "$ADAPTER"
```

When running, type a ticket file path at the prompt
(e.g. `demo_tickets/it_password_reset.txt`) and select a prompt (P1–P5).

### Save code changes to GitHub

```python
import subprocess, os
BRANCH = 'pair-1/smollm2-1.7b'  # ← your branch
MSG    = 'Describe your changes'  # ← update this

for cmd in [['git','add','.'], ['git','commit','-m',MSG], ['git','push','origin',BRANCH]]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd='/content/project')
    if (r.stdout + r.stderr).strip():
        print((r.stdout + r.stderr).strip())
```

---

## Recommended experiments

Run in this order. Each one builds on the last.

| Model | `model_id` in config | Epochs | T4 training time |
|-------|---------------------|--------|-----------------|
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M-Instruct` | 3 | ~10 min |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 5 | ~20 min |
| Phi-3.5-Mini ★ | `microsoft/Phi-3.5-mini-instruct` | 5 | ~30 min |
| Llama-3.2-8B | `meta-llama/Llama-3.2-8B-Instruct` | 3 | ~90 min |

★ Primary Phase 1 model. Start here for your first real experiment.

**Enable BERTScore on Colab** (works on Colab's Python 3.12):
```yaml
evaluation:
  run_bertscore: true
  bertscore_model: "roberta-large"
```

---

## Colab limits

| Limit | Detail |
|-------|--------|
| GPU | T4 (16 GB) — not guaranteed. Reconnect if you get K80 (12 GB). |
| Session length | Up to 12 hours — training always finishes well within this |
| Idle timeout | ~90 min with no active code — sessions stay alive during training |
| Weekly quota | ~15–30 GPU hours — enough for 3–5 full experiments per week |

---

## Where files live

| What | Location | Persists? |
|------|----------|-----------|
| Code | GitHub (your branch) | ✅ Push to save |
| Labeled CSVs, checkpoints | Drive: `data/` | ✅ Always |
| LoRA adapters, eval results | Drive: `outputs/{run_id}/` | ✅ Always |
| HuggingFace model cache | Drive: `hf_cache/` | ✅ First download only |

---

## Resuming after a session ends

Training checkpoints save after each epoch to Drive. To resume:

1. Run the setup notebook (Cells 1–7)
2. In `configs/phase1_config.yaml`, set the steps that already completed to `false`
3. Re-run the pipeline — it picks up from where it left off

---

## Common issues

| Problem | Fix |
|---------|-----|
| Cell 6 shows ❌ for a secret | Open 🔑 Secrets → find the secret → toggle Notebook access ON (blue) |
| Cell 7 fails "Code directory not found" | Cell 3 didn't run successfully — re-run Cell 3 |
| GPU shows K80 | Runtime → Disconnect and delete runtime → reconnect until you get T4 |
| CUDA out of memory | Set `per_device_train_batch_size: 2` in config |
| Session ended mid-training | Restart, run setup notebook, set completed steps to `false` in config, re-run |
