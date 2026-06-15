from huggingface_hub import HfApi, HfFolder, Repository
from transformers import AutoTokenizer, AutoModelForCausalLM

# Your username or organization + desired repo name
repo_id = "Jiajunruan/Minmax-TOFU-2"  # CHANGE THIS
# repo_id = "Jiajunruan/MUSE-News_RULE-NPO2"
# repo_id = "Jiajunruan/Minmax-TOFU"
local_path = "/u/jiajunr2/Minmax_Probe/TOFU/baselines/output"
# local_path = "/users/2/jruan/training_muse/baselines/ckpt/news/npo/iter_3"
# local_path = "/users/2/jruan/Probe_unlearning/output"

# # Create the repo on HF Hub (private if needed)
from huggingface_hub import create_repo
create_repo(repo_id="Jiajunruan/Minmax-TOFU-2", exist_ok=True)

# Clone and push
from huggingface_hub import upload_folder
upload_folder(
    repo_id=repo_id,
    folder_path=local_path,
    path_in_repo=".",
    commit_message="Initial commit of unlearned LLaMA model",
)