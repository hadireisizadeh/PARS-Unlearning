from huggingface_hub import HfApi, HfFolder, Repository
from transformers import AutoTokenizer, AutoModelForCausalLM

# Your username or organization + desired repo name
# repo_id = "Jiajunruan/Prism-MUSE"  # CHANGE THIS
repo_id = "Jiajunruan/MUSE-News_RULE-NPO2"
# repo_id = "Jiajunruan/Minmax-TOFU"
# local_path = "/users/2/jruan/Probe_unlearning_muse/baselines/output"
local_path = "/work/nvme/bhdx/jiajunr2/PARS/MUSE/lr_2e-5_epochs_10_probe_layers_6_20_25_28_beta_0.5_r_f_both"
# local_path = "/users/2/jruan/Probe_unlearning/output"
# local_path = "/work/nvme/bhdx/jiajunr2/prism"

# # Create the repo on HF Hub (private if needed)
# from huggingface_hub import create_repo
# create_repo(repo_id, private=True, exist_ok=True)

# Clone and push
from huggingface_hub import upload_folder
upload_folder(
    repo_id=repo_id,
    folder_path=local_path,
    path_in_repo=".",
    commit_message="Initial commit of unlearned LLaMA model",
)