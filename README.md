# Mask-Free Privacy Extraction and Rewriting: A Domain-Aware Approach via Prototype Learning

---

## Preliminaries

### 1) Environment Setup

The experiments are conducted in **two independent runtime environments**.  
To ensure reproducibility, please install the required Python dependencies **separately** using the following files:

- `requirements_damper.txt`
- `requirements_llamafactory.txt`

> **Note:** We recommend creating two isolated environments (e.g., using Conda) to avoid dependency conflicts.

---

### 2) Dataset Preparation

Please **manually download** the following datasets:

- **Pri-DDXPlus**
- **Pri-SLJA**

After downloading, split the datasets according to the experimental setup and place the processed files under the `datasets/` directory.

---

### 3) Pre-trained Models

Download all required pre-trained models and place them under the `models/` directory.

---

## Quick Start

### Step 1: Prepare Training Data for Contrastive Learning

Generate the prototype-based training dataset by running:

```bash
python create_prototype_dataset.py
```

### Step 2: Run the Full Training and Evaluation Pipeline

Execute the end-to-end pipeline using the following command:

```bash
./automatic.sh \
  -d 0 \
  -t 0.1 \
  -a 0.3 \
  -b 0.1 \
  -g1 0.85 \
  -g2 0.8 \
  -e 150 \
  --from-step 1 \
  --end-step 13
```