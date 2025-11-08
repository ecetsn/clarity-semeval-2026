# CLARITY-SemEval-2026  
### Two-Stage Evasion-to-Clarity Modeling for Ambiguity Detection in Question–Answer Interactions

![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)

---

## Overview

This repository contains our full pipeline for the **CLARITY (SemEval-2026)** shared task:
- **Task 1:** Predict whether a response to an interview question is *Clear Reply*, *Ambiguous*, or *Clear Non-Reply*.  
- **Task 2:** Detect the *evasion technique* used in a response (9-class taxonomy).

We combine **instruction-tuned large language models** with **linguistically informed preprocessing** to reach state-of-the-art Macro-F1 on both tasks.

The system extends ideas from the CLARITY benchmark paper ([Papantoniou et al., 2024](https://arxiv.org/abs/2409.13879)) and improves them through:
- A **two-step evasion→clarity pipeline** using LoRA fine-tuning.  
- **Named-entity scrubbing** and **question decomposition** to handle long and multi-part inputs.  
- **Dual-pass prompting** (full vs grounded) and **ensemble calibration** for stable predictions.

---

## Planned Repository Structure

```
clarity-semeval-2026/
│
├── config/ # YAML configs for tasks and LoRA params
├── data/ # raw / processed / cache datasets
├── src/
│ ├── data/ # preprocessing, loaders, augmentation
│ ├── prompts/ # template prompts & few-shot examples
│ ├── models/ 
│ ├── training/ 
│ ├── evaluation/ # metrics, confusion, ensemble
│ └── utils/ # logging, config, seed, HF helpers
│
├── notebooks/ # analysis
├── experiments/ # logs and saved runs
└── reports/ # figures, results
```