"""
Tournament script for GRF_MARL - Round Robin between all pre-trained 5v5 models.
Records videos and outputs results.
"""

import os
import pathlib
import sys
import itertools

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE_DIR)

import numpy as np
from light_malib.rollout.rollout_func import rollout_func
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.envs.gr_football.env import GRFootballEnv
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.utils.cfg import load_cfg
from light_malib.utils.logger import Logger

# Configuration
MODELS_DIR = os.path.join(BASE_DIR, "light_malib", "trained_models", "gr_football", "5_vs_5")
CONFIG_PATH = os.path.join(BASE_DIR, "expr_configs", "cooperative_MARL_benchmark", "full_game", "5_vs_5_hard", "ippo.yaml")
TOURNAMENT_LOG_DIR = os.path.join(BASE_DIR, "tournament_logs")

os.makedirs(TOURNAMENT_LOG_DIR, exist_ok=True)

# Get model names
model_names = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
print(f"Found {len(model_names)} models: {model_names}")

# Load config
cfg = load_cfg(CONFIG_PATH)

# Enable rendering and video recording
cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["render"] = True
cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["write_full_episode_dumps"] = True
cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["write_video"] = True
cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["logdir"] = TOURNAMENT_LOG_DIR

# Create environment
env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

def load_policy(model_name, agent_id):
    path = os.path.join(MODELS_DIR, model_name)
    try:
        policy = MAPPO.load(path, env_agent_id=agent_id)
        return policy
    except Exception as e:
        Logger.error(f"Failed to load {model_name}: {e}")
        return None

# Results storage
results = []

# Round robin
matches = list(itertools.combinations(model_names, 2))
print(f"Running {len(matches)} matches...")

for i, (model_a, model_b) in enumerate(matches):
    print(f"\n=== Match {i+1}/{len(matches)}: {model_a} vs {model_b} ===")
    
    policy_a = load_policy(model_a, "agent_0")
    policy_b = load_policy(model_b, "agent_1")
    
    if policy_a is None or policy_b is None:
        print("Skipping match due to load failure")
        results.append({"model_a": model_a, "model_b": model_b, "winner": "ERROR", "score": "N/A"})
        continue
    
    behavior_policies = {
        "agent_0": ("policy_a", policy_a),
        "agent_1": ("policy_b", policy_b),
    }
    
    try:
        rollout_results = rollout_func(
            eval=True,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=3000,
            render=True,
            rollout_epoch=0,
        )
        
        stats_a = rollout_results["results"][0]["stats"]["agent_0"]
        score_a = stats_a.get("score", 0)
        score_b = stats_a.get("lost", 0)
        
        if score_a > score_b:
            winner = model_a
        elif score_b > score_a:
            winner = model_b
        else:
            winner = "Draw"
        
        print(f"Result: {model_a} {score_a} - {score_b} {model_b} | Winner: {winner}")
        results.append({"model_a": model_a, "model_b": model_b, "winner": winner, "score": f"{score_a}-{score_b}"})
        
    except Exception as e:
        Logger.error(f"Match failed: {e}")
        results.append({"model_a": model_a, "model_b": model_b, "winner": "ERROR", "score": str(e)})

# Print final results
print("\n" + "="*60)
print("TOURNAMENT RESULTS")
print("="*60)

wins = {}
for r in results:
    if r["winner"] not in ["Draw", "ERROR"]:
        wins[r["winner"]] = wins.get(r["winner"], 0) + 1

print("\nWin Counts:")
for model, w in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {model}: {w} wins")

print(f"\nAll match results saved. Videos in: {TOURNAMENT_LOG_DIR}")
