# Domain-Specific Theme Labeling via Parameter-Efficient Fine-Tuning (PEFT) on SLM

**Company / Org:** Automation Anywhere  
**Challenge Advisor:** Archan Dutta, archanduttads@gmail.com  
**AI Studio Coach:** Shaun Figueiro, shaun.figueiro@breakthroughtech.org          
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Automation Anywhere
Automation Anywhere is a global leader in Intelligent Automation, specializing in Robotic Process Automation (RPA) and AI-driven business solutions. The company empowers organizations to scale their operations through digital transformation, focusing on deploying intelligent software bots to handle complex business processes efficiently.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use datasets (preferably in IT/HR/CX) and ML technique named distillation (using LLMs to teach SLMs) to label topics and then compare the performance of the LLM and SLM. In Enterprise AI, the Distilled SLM may result in a model that matches LLM-level accuracy at lower cost and lower latency. One way to do so is by using Parameter-Efficient Fine-Tuning (PEFT) - LoRA or QLoRA.

### Success Criteria
- SLM achieves reasonable accuracy on theme labeling when compared to Frontier LLM.
- SLM reduces cost-per-inference when compared to Frontier LLM.
- SLM reduces inference latency when compared to Frontier LLM.
- Demo: real-time side-by-side output comparing LLM vs. SLM on live ticket input.

NOTE: The Success metrics could change as the team converges on the project scope.

### Stretch Goals
Based on the results of the experiments, write a Research Paper.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Duration | Milestone | Key Activities |
|---------------|---------------|----------------|
| **2 weeks - Ends on Sep 15th** | Business Problem and Code Setup | Understand the Business Problem. <br> High-level Project Plan. <br> Team Task Distribution. <br> Environment and Code Setup - Run Smoke Test. |
| **6 weeks - End on Oct 31st** | Planning, Fine-Tuning SLMs and Experimentation | Detailed Project Planning. <br> Understanding the Pipeline.  <br> Run Fine-Tuning Experiments on Smaller Dataset. <br> Run Fine-Tuning Experiments on Larger Dataset.  <br> Change Implementation as needed - LoRA, QLoRA, Prompt Engineering, Metrics etc.  <br> Evaluation of experiments. |
| **2 weeks - Ends on Nov 15th** | Analysis, Findings and Discussion | Analyze the results of experiments and highlight observations and findings. <br> Discussion on Performance Dimension - Latency, Cost, Quality, Generalizability.|
| **2 weeks - End of Nov 31st** | Storytelling and Presentation | Consolidate Results. <br> Internal Demo. <br> Prepare Presentation for BTT Demo |

NOTE: The Milestones could change as the team converges on the project scope.

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** bitext/Bitext-customer-support-llm-chatbot-training-dataset    
**Format:** CSV/ TSV,JSON,Parquet,Excel (.xlsx)     
**Size:** 1gb to 5gb  
**Location:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

### Learnings from this Project
Students will be exposed to a lot of essential AI/ML techniques:
- Parameter-Efficient Fine-Tuning (PEFT) - LoRA or QLoRA.
- Evaluation Techniques
- Running SLMs locally, or on Google Colab. Google Colab is highly preferred as it will allow fair comparison between models.
- Learn about multiple open source SLMs
- Solve Technical Problems to improve Business metrics
- Improve Communication and Teanwork

---

## 🛠️ Suggested Approach

**ML Problem Type:** Natural Language Processing (NLP), Large Language Models (LLMs)/ Generative AI / Parameter-Efficient Fine-Tuning / Distillation / 

**Recommended Libraries:**
- pandas, scikit-learn, torch, Hugging Face, sentence-transformers, openai, anthropic, bitsandbytes, unsloth etc.

**Evaluation Metrics:**
- Relevance, Equivalence, Specificity, ROUGE, BERTScore etc.
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- What is LLM-as-a-Judge ?  https://www.youtube.com/watch?v=dQWzKeifcFs
- LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide   https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation.  https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

**Technical Tutorials:**
- What is LLM Distillation?   https://www.youtube.com/watch?v=h7DUpHPasME
- LLM Distillation Explained: Applications, Implementation & More   https://www.datacamp.com/blog/distillation-llm
- LLM (Parameter Efficient) Fine Tuning - Explained!   https://www.youtube.com/watch?v=HcVtpLAGMXo
- LoRA & QLoRA Fine-tuning Explained In-Depth.  https://www.youtube.com/watch?v=t1caDsMzWBk

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Your team's channel within Break Through Tech’s Discord space.
* Email Please copy your teammates and AI Studio Coach. Here's my email: archanduttads@gmail.com 
* Request a team check-in on Zoom on Google Meet.
* Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.
* Availability:  Tuesday, Wednesday, Thursday, anytime between 7 pm and 8 pm PST

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

---

## 🚀 Getting Started

1. **Clone the Github Repo and follow the README instructions - For Local Setup and For Google Colab Setup
2. **Review the code and note any questions for our first meeting.
3. **Begin reviewing the dataset** using the link above.
4. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
