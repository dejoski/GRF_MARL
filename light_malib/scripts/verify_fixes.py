
import numpy as np
import torch
import os
import pathlib
import sys

# Add parent dir to path to find light_malib
BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent)
sys.path.append(BASE_DIR)

from light_malib.envs.gr_football.env import GRFootballEnv
from light_malib.algorithm.dqn.policy import DQN
from light_malib.utils.cfg import load_cfg
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.rollout.rollout_func import rollout_func
from light_malib.buffer.data_server import DataServer
from light_malib.utils.naming import default_table_name
from light_malib.algorithm.mappo.policy import MAPPO

config_path = "expr_configs/cooperative_MARL_benchmark/full_game/5_vs_5_hard/ippo.yaml"
cfg = load_cfg(os.path.join(BASE_DIR, config_path))

print("Initializing Environment...")
env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])

print("Initializing DQN Policy...")
model_config = cfg.populations[0].algorithm.model_config
custom_config = cfg.populations[0].algorithm.custom_config
model_config["model"] = "gr_football.basic"
custom_config["use_cuda"] = False 

# Use MAPPO's logic for spaces (DQN handles it internally via FeatureEncoder now)
# We pass placeholder spaces, DQN overwrites observation_space with encoder's
dqn_policy = DQN("DQN", None, 19, model_config, custom_config) # 19 actions

print("DQN Policy Initialized.")

print("Running Rollout...")
behavior_policies = {
    "agent_0": ("policy_0", dqn_policy),
    "agent_1": ("policy_1", dqn_policy) 
}
dataserver = DataServer('dataserver_1', cfg.data_server)
rollout_desc = RolloutDesc("agent_0", "policy_0", None, None, None, None)
table_name = default_table_name(rollout_desc.agent_id, rollout_desc.policy_id, None)
dataserver.create_table(table_name)

res = rollout_func(
    eval=False,
    rollout_worker=None,
    rollout_desc=rollout_desc,
    env=env,
    behavior_policies=behavior_policies,
    data_server=dataserver,
    rollout_length=10,
    sample_length=0,
    render=False,
    rollout_epoch=0
)
print("Rollout finished.")

print("Checking Data Server for Active Mask...")
sample, _ = dataserver.sample(table_name, batch_size=5)
# sample is a list of dicts or dict of lists depending on impl. data_server.sample returns (data, info)
# Checking structure
if isinstance(sample, list) and len(sample) > 0:
    first_item = sample[0]
    if EpisodeKey.ACTIVE_MASK in first_item:
        print("SUCCESS: Active Mask found in sample!")
        try:
            print("Shape:", first_item[EpisodeKey.ACTIVE_MASK].shape)
        except:
            print("Val:", first_item[EpisodeKey.ACTIVE_MASK])
    else:
        print("FAILURE: Active Mask NOT found in sample.")
        print("Keys:", first_item.keys())
else:
    print("FAILURE: No sample returned or unexpected format.")
    print("Sample:", sample)

print("Verification Complete.")
