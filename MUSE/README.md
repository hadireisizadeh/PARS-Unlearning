PARS: Hidden state unlearning

## 🛠️ Installation

### Conda Environment

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate py310
```


## 🚀 Quick Start: Run Probe via SLURM

To run the probe unlearning baseline on Delta:

```bash
cd baselines
sbatch unlearn_news.sh
```

Before submitting, make sure to update the following in `submit_muse.sh`:
- `--mail-user`: Set to your email address
- `OUT_BASE`: Set to your output directory path

The script runs `probe_npo_gdr` with probe layers and evaluates using `eval.py`.

---
