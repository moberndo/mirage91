import os

import numpy as np
import wandb
import torch
import random

import yaml

from offline_scripts.classifier_LMDA_PhS import evaluate

CONFIG_PATH = "./configs/"

wandb.login()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_weights_and_bias(config: dict):
    if config["wandb_disabled"]:
        mode = "offline"
    else:
        mode = "online"
    wandb.init(
        mode=mode,
        dir="./classifier_results",
        project="mirage91",
        config=config,
    )


def run_evaluation(config: dict):
    init_weights_and_bias(config=conf)
    evaluate(device=device, config=config)

if __name__ == "__main__":
    try:
        conf = load_config("config.yaml")
        run_evaluation(conf)
    except Exception as e:
        print(f"Enter as second argument a valid path to the config file.\nDetailed error message: {e}")