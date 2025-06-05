from __future__ import annotations
import argparse
import time
import wandb
import torch
import gymnasium as gym
import numpy as np
import os

from env.custom_hopper import *
from agent import Agent, Policy, Critic
from gymnasium.vector import AsyncVectorEnv


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=100_000, help="# episodi di training")
    p.add_argument("--print_every", type=int, default=20_000, help="log ogni N episodi")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument('--n_envs', default=1, type=int, help='Select number of training envs')
    p.add_argument("--render", action="store_true", help="visualizza la GUI durante il training")

    p.add_argument('--domain', default='source', choices=["source", "target"], help="Domain to train on [source, target]")
    p.add_argument("--WandDB", action="store_true", help="Use WandDB Callback")
    p.add_argument("--save", action="store_true", help="Save the model")

    p.add_argument('--AC_critic', default = 'Q', type=str, choices = ['Q', 'V'], help="Whether critic estimates Q(s,a) or V(s)")

    p.add_argument("--gamma", default=0.99, type = float, help="Gamma to discount future rewards")
    p.add_argument("--lr_policy", default=5e-4, type = float, help="Learning Rate for Policy Updates")
    p.add_argument("--lr_critic", default=5e-4, type = float, help="Learning Rate for Critic Updates")
    p.add_argument("--hidden", default=64, type=int, help='Hidden Layers for Critic and Actor Nets')
    return p.parse_args()

args = parse_args()

# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    if args.WandDB:
        run_name = f"AC_{args.domain}_g{args.gamma}_lrp{args.lr_policy}_lrc{args.lr_critic}_h{args.hidden}"
    
        wandb.init(
            project="Reinforcement Learning",
            config={
                "domain": args.domain,
                "n_episodes": args.n_episodes,
                "device": args.device,
                "gamma": args.gamma,
                "lr_policy": args.lr_policy,
                "lr_critic": args.lr_critic,
                "hidden": args.hidden,
                "AC_critic": args.AC_critic,
            },
            name=run_name,
            sync_tensorboard=True,
            monitor_gym=True,
        )

    def make_env(seed_offset):
        def _init():
            env = gym.make(f"CustomHopper-{args.domain}-v0")  # oppure "CustomHopper-target-v0"
            env.reset(seed=seed_offset)
            return env
        return _init

    env = AsyncVectorEnv([make_env(i) for i in range(args.n_envs)])

    def save(policy, critic, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        model_path = os.path.join(dir_path, f"best_model_{args.domain}")
        os.makedirs(model_path, exist_ok=True)
        policy_path = os.path.join(model_path, f"policy.mdl")
        torch.save(policy.state_dict(), policy_path)
        critic_path = os.path.join(model_path, f"critic.mdl")
        torch.save(critic.state_dict(), critic_path)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    max_action = env.action_space.high

    args_policy = {
        "state_space" : obs_dim,
        "action_space" : act_dim,
        "hidden" : args.hidden
        }
    
    args_critic = {
        "state_space" : obs_dim,
        "action_space" : act_dim,
        "hidden" : args.hidden
    }
    
    args_agent = {
        "max_action" : max_action,
        "device" : args.device,
        "gamma" : args.gamma,
        "lr_policy" : args.lr_policy,
        "lr_critic" : args.lr_critic,
        "AC_critic" : args.AC_critic
    }

    def train():
        if args.AC_critic == 'V':
            args_critic["action_space"] = 0 

        args_agent["policy"] = Policy(**args_policy)
        args_agent["critic"] = Critic(**args_critic)
        agent = Agent(**args_agent)
        best_score = env.envs[0].reward_range[0]
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
                        agent.store_outcome(prev_obs[i], obs[i], log_probs[i], rewards[i], done[i])
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
                    save(agent.policy, agent.critic, dir_path)
            
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
                elapsed = time.time() - start_time
                print(f"[Episode {episode+1}] mean_return = {ep_returns.mean():.2f} ± {ep_returns.std():.2f}, loss = {loss:.4f}, elapsed = {elapsed:.1f}s")
                start_time = time.time()
        
        print(f"FINAL_RESULT: {ep_returns.mean():.2f} ± {ep_returns.std():.2f}")
        print(f"BEST_MODEL_AVG_SCORE: {best_score:.2f} ± {best_std_score:.2f}")

        if args.save:
            final_model_path = os.path.join('models', f"last_model_{args.domain}")
            os.makedirs(final_model_path, exist_ok=True)
            torch.save(agent.policy.state_dict(), os.path.join(final_model_path, "policy.mdl"))
            torch.save(agent.critic.state_dict(), os.path.join(final_model_path, "critic.mdl"))

        env.close()

    train()
if __name__ == "__main__":
    main()
