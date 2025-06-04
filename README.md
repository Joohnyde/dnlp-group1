# 🧠 LoRA: Reproduction and Extension Challenge

This repository contains the work of **Group 1** for the **Reproducibility and Extension Challenge** assigned in the course:

> **192.039 Deep Natural Language Processing**  
> Vienna University of Technology (TU Wien) · Summer Semester 2025

We focus on the paper:

> **LoRA: Low-Rank Adaptation of Large Language Models**  
> Edward Hu, Yelong Shen, Phillip Wallis, et al.  
> [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## 🎯 Project Objectives

The goal of this project was to:
- 📖 **Understand** the motivation and mechanics of LoRA.
- 🔁 **Reproduce** a key experiment from the paper (RoBERTa fine-tuning on SST-2).
- 🧪 **Extend** the original work by testing LoRA on additional setups and parameters.
- 🧑‍🏫 **Present** our findings in a detailed presentation and participate in group interviews.

---

## 👥 Group Members

| Name              | Student ID   |
|-------------------|--------------|
| Denijal Sendelj   | 12129474     |
| Richard Keil      | 12447214     |
| Thomas Ranner     | 11771213     |

---

## 📌 Summary of the Paper

**LoRA** introduces a simple, scalable technique for efficient fine-tuning of pre-trained large language models (LLMs).  
Instead of updating all model parameters, LoRA freezes them and injects trainable low-rank matrices into specific layers (e.g., attention projections). This reduces the number of trainable parameters by orders of magnitude, while achieving comparable performance to full fine-tuning.

Key contributions:
- Efficient parameter-efficient fine-tuning (PEFT).
- Strong empirical results across several models (RoBERTa, GPT-2, etc.).
- Reusable and model-agnostic method.

---

## 🛠️ Codebase Contents

dnlp-group1/ 

├── reproduction/ # Reproduction of the paper's experiment 

│ ├── train_baseline.py # Full fine-tuning (baseline) 

│ ├── train_lora.py # LoRA fine-tuning using PEFT 

│ └── utils.py

├── extension/ # Extensions beyond the original paper 

│ ├── alt_task_mnli.py # LoRA on a new GLUE task (e.g. MNLI) 

│ ├── low_rank_analysis.py # Varying LoRA rank and analyzing performance 

│ └── no_attention_test.py # Experimenting with non-standard modules 

├── plots/ # Accuracy, parameter count, training graphs 

├── notebooks/ # Exploratory or evaluation notebooks 

├── presentation/ # Final PowerPoint slides 

│ └── LoRA_TUW_Group1.pdf 

├── requirements.txt # Python dependencies 

├── README.md # This file 

└── LICENSE

---

## 🔁 Reproduction

We reproduced the fine-tuning of **RoBERTa-base** on the **SST-2** sentiment classification task, comparing:

- **Full fine-tuning** (all weights updated)
- **LoRA fine-tuning** (only low-rank matrices updated)

We used:
- HuggingFace Transformers
- HuggingFace Datasets (GLUE benchmark)
- PEFT library from HuggingFace for LoRA integration

📊 Results:

| Method           | Accuracy (SST-2) | Trainable Params | Training Time |
|------------------|------------------|------------------|----------------|
| Full Fine-Tuning | 93.2%            | ~125M            | ~12 min        |
| LoRA (r=8)       | 92.9%            | ~1.8M            | ~9 min         |

✅ LoRA nearly matches full fine-tuning performance with only ~1.5% of the trainable parameters.

---

## 🧪 Extensions

We conducted several novel experiments to extend the findings of the paper:

### 🔹 1. LoRA on a Different Task: MNLI
We tested LoRA on a more complex GLUE task – **Multi-Genre Natural Language Inference (MNLI)** – to see how it generalizes beyond SST-2.

### 🔹 2. LoRA Rank Sweep
We ran a series of experiments with different **LoRA ranks (r = 1, 2, 4, 8, 16)** to observe the trade-off between parameter count and accuracy.

### 🔹 3. Non-Attention Target Modules (Exploratory)
We tried applying LoRA to modules **outside of attention**, such as feed-forward layers, to test how sensitive the technique is to where it’s applied.

All extensions are fully documented in the `extension/` folder and discussed in the final presentation.

---

## 📽️ Presentation

You can find our final presentation slides here:  
📄 [`presentation/LoRA_TUW_Group1.pdf`](./presentation/LoRA_TUW_Group1.pdf)

Presentation structure:
1. **Introduction** – Problem space, what LoRA is
2. **Understanding** – Theory and intuition
3. **Reproduction** – Code, setup, results
4. **Extensions** – Our new experiments
5. **Conclusion** – Takeaways, reproducibility notes, future work

---

## 🧠 Individual Contributions

- **Denijal Sendelj**: Reproduced SST-2 experiments, implemented full vs. LoRA runs, plotted training graphs.
- **Richard Keil**: Designed and implemented LoRA-on-MNLI extension and parameter sweep.
- **Thomas Ranner**: Handled theoretical explanation, slides, and exploratory experiments on LoRA in feed-forward layers.

Each team member also participated in group debugging, syncs, and presentation preparation.

---

## 🔗 Acknowledgments

- PEFT Library: https://github.com/huggingface/peft  
- Transformers: https://github.com/huggingface/transformers  
- Datasets: https://huggingface.co/datasets/glue

> All reused code is acknowledged and modified appropriately. Our extensions go beyond publicly available implementations.

---

## 📜 License

This project is licensed under the MIT License.  
See [LICENSE](./LICENSE) for details.

---

