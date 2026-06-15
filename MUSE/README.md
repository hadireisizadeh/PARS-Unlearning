# <img alt="icon" src="figures/icon.png" height=30> PARS: Hidden state unlearning



## 🛠️ Installation

### Conda Environment

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate py310
```

## 📘 Data & Target Models

Two corpora `News` and `Books` and the associated target models are available as follows:

| Domain | <div style="text-align: center">Target Model for Unlearning</div> | Dataset |
|----------|:------------------------------:|----------| 
| News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
| Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) | 

Before proceeding, load all the data from HuggingFace to the root of this repostiory by running the following instruction:
```
python load_data.py
```

## 🚀 Quick Start: Run MinMax Probe via SLURM

To run the minimax probe unlearning baseline on Delta:

```bash
cd baselines
sbatch submit_muse.sh
```

Before submitting, make sure to update the following in `submit_muse.sh`:
- `--mail-user`: Set to your email address
- `OUT_BASE`: Set to your output directory path

The script runs `minimax_npo_gdr` with probe layers and evaluates using `eval.py`.

---
