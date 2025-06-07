from __future__ import annotations
import argparse
import time
import wandb
import torch
import gymnasium as gym
import numpy as np
import os

from env.custom_hopper import *
from agent import Agent, Policy, BaselineMLP
from gymnasium.vector import AsyncVectorEnv

# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=100_000, help="number of training episodes")
    p.add_argument("--print_every", type=int, default=20_000, help="log every N episodes")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument('--n_envs', default=1, type=int, help='Select number of training envs for parallel training')
    p.add_argument("--render", action="store_true", help="show GUI during training")

    p.add_argument('--domain', default='source', choices=["source", "target"], help="Domain to train on [source, target]")
    p.add_argument("--WandDB", action="store_true", help="Use WandDB Callback")
    p.add_argument("--save", action="store_true", help="Save the model")

    p.add_argument('--baseline', default='0', type=str, help="Insert a value or write 'dynamic' to state-dependent baseline")

    p.add_argument("--gamma", default=0.99, type = float, help="Gamma to discount future rewards")
    p.add_argument("--lr_policy", default=5e-4, type = float, help="Learning Rate for Policy Updates")
    p.add_argument("--lr_baseline", default=5e-4, type = float, help="Learning Rate for BaselineMLP Updates")
    p.add_argument("--hidden", default=64, type=int, help='Hidden Layers for BaselineMLP and Actor Nets')

    return p.parse_args()

args = parse_args()

# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:

    if args.WandDB:
        run_name = f"REINFORCE_{args.domain}_g{args.gamma}_lrp{args.lr_policy}_lrb{args.lr_baseline}_bsln{args.baseline}_h{args.hidden}_ep{args.n_episodes}"

        wandb.init(
            project="Reinforcement Learning",
            config={
                "domain": args.domain,
                "n_episodes": args.n_episodes,
                "device": args.device,
                "gamma": args.gamma,
                "lr_policy": args.lr_policy,
                "lr_critic": args.lr_baseline,
                "baseline": args.baseline,
                "hidden": args.hidden,
            },
            name=run_name,
            sync_tensorboard=True,
            monitor_gym=True,
        )

    def make_env(seed_offset):
        def _init():
            env = gym.make(f"CustomHopper-{args.domain}-v0")
            env.reset(seed=seed_offset)
            return env
        return _init

    env = AsyncVectorEnv([make_env(i) for i in range(args.n_envs)])

    def save(policy, baseline_model, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        model_path = os.path.join(dir_path, f"best_model_{args.domain}_{args.baseline}baseline")
        os.makedirs(model_path, exist_ok=True)
        policy_path = os.path.join(model_path, "policy.mdl")
        torch.save(policy.state_dict(), policy_path)
        if baseline_model is not None:
            baseline_path = os.path.join(model_path, "baseline.mdl")
            torch.save(baseline_model.state_dict(), baseline_path)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    max_action = env.action_space.high

    args_policy = {
        "state_space" : obs_dim,
        "action_space" : act_dim,
        "hidden" : args.hidden
        }
    
    args_baselineMLP = {
        "state_space" : obs_dim,
        "hidden" : args.hidden
    }
    
    args_agent = {
        "max_action" : max_action,
        "device" : args.device,
        "gamma" : args.gamma,
        "lr_policy" : args.lr_policy,
        "lr_baseline" : args.lr_baseline,
        "baseline" : args.baseline,
        "n_envs": args.n_envs,
    }

    def train():
        args_agent["policy"] = Policy(**args_policy)

        if args.baseline == "dynamic":
            args_agent["baseline_model"] = BaselineMLP(**args_baselineMLP)
            args_agent["baseline"] = 0
        else:
            args_agent["baseline_model"] = None
            args_agent["baseline"] = float(args.baseline)
        
        agent = Agent(**args_agent)

        best_score = -np.inf
        score_history = []

        start_time = time.time()
        for episode in range(args.n_episodes):
            obs, _ = env.reset(seed=episode)
            terminated = [False] * env.num_envs
            ep_returns = np.zeros(env.num_envs)

            while not all(terminated):
                actions, log_probs = agent.get_action(obs)
                prev_obs = obs
                obs, rewards, term, trunc, _ = env.step(actions)
                done = np.logical_or(term, trunc)

                for i in range(env.num_envs):
                    if not terminated[i]:
                        agent.store_outcome(i, prev_obs[i], log_probs[i], rewards[i], done[i])
                        ep_returns[i] += rewards[i]
                terminated = np.logical_or(terminated, done)

            loss, adv_std = agent.update_policy()
            score_history.append(np.mean(ep_returns))
            if len(score_history) >= 100:
                avg_score = np.mean(score_history[-100:])
                avg_std_score = np.std(score_history[-100:])
            else:
                avg_score = np.mean(score_history)
                avg_std_score = np.std(score_history)

            if avg_score > best_score:
                best_score = avg_score
                best_std_score = avg_std_score
                if args.save:
                    dir_path = 'models'
                    save(agent.policy, agent.baseline_model, dir_path)

            if args.WandDB:
                wandb.log({
                    "episode": episode + 1,
                    "mean_return": ep_returns.mean(),
                    "advantage_std": adv_std,
                    "reward_std_window": avg_std_score,
                    "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                    "time_elapsed": time.time() - start_time,
                })

            if args.render:
                env.render()

            if (episode + 1) % args.print_every == 0:
                print(f"[Episode {episode+1}] return = {ep_returns.mean():.2f}, loss = {loss:.4f}")
                
        print(f"FINAL_RESULT: {ep_returns.mean():.2f} ± {ep_returns.std():.2f}")
        print(f"BEST_MODEL_AVG_SCORE: {best_score:.2f} ± {best_std_score:.2f}")

        if args.save:
            final_model_path = os.path.join('models', f"last_model_{args.domain}_{args.baseline}baseline")
            os.makedirs(final_model_path, exist_ok=True)
            torch.save(agent.policy.state_dict(), os.path.join(final_model_path, "policy.mdl"))
            if agent.baseline_model is not None:
                torch.save(agent.baseline_model.state_dict(), os.path.join(final_model_path, "baseline.mdl"))
        env.close()

    train()
if __name__ == "__main__":
    main()
