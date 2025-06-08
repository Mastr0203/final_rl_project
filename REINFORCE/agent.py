import numpy as np
import torch
from torch.distributions import Normal

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden=64):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        init_sigma = 0.5
        init_log_sigma = np.log(init_sigma)
        self.log_sigma = torch.nn.Parameter(torch.full((action_space,), init_log_sigma, dtype=torch.float32))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.tanh(self.fc1_actor(x))
        x = self.tanh(self.fc2_actor(x))
        action_mean = self.fc3_actor_mean(x)

        sigma = torch.exp(self.log_sigma).expand(x.size(0), -1).clamp(min=1e-4, max=1.0)

        return Normal(action_mean, sigma)

class BaselineMLP(torch.nn.Module):
    """
    BaselineMLP estimates the state-value function V(s) given only the state as input.
    It outputs a scalar representing the expected discounted return from state s under the current policy.
    """
    def __init__(self, state_space, hidden=64):
        super().__init__()
        self.state_space = state_space
        self.hidden = hidden
        self.tanh = torch.nn.Tanh()

        self.fc1_baseline = torch.nn.Linear(state_space, self.hidden)
        self.fc2_baseline = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_baseline_mean = torch.nn.Linear(self.hidden, 1)
        self.init_weights()

    def forward(self, state):
        """
        Forward pass for BaselineMLP.
        Args:
            state: Tensor of shape [batch_size, state_space]
        Returns:
            Tensor of shape [batch_size, 1] with the estimated V(s).
        """
        x = self.tanh(self.fc1_baseline(state))
        x = self.tanh(self.fc2_baseline(x))
        return self.fc3_baseline_mean(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

class Agent(object):
    def __init__(self, policy: Policy, max_action, baseline_model: BaselineMLP | None = None, device: str = 'cpu',
        gamma: float = 0.99,
        lr_policy: float = 5e-4,
        lr_baseline: float = 5e-4,
        baseline: float = 0,
        n_envs: int = 1,
    ):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.baseline_model = baseline_model.to(self.train_device) if baseline_model is not None else None
        self.max_action = torch.tensor(max_action, device=self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        if self.baseline_model is not None:
            self.optimizer_baseline = torch.optim.Adam(self.baseline_model.parameters(), lr=lr_baseline)
        else:
            self.optimizer_baseline = None
        
        # --- Buffers ---
        self.gamma = gamma
        self.baseline = baseline
        self.n_envs = n_envs
        # To explain
        self.states_buffer = [[] for _ in range(n_envs)]
        self.action_log_probs_buffer = [[] for _ in range(n_envs)]
        self.rewards_buffer = [[] for _ in range(n_envs)]
        # store completed episodes
        self.episode_data = [] # To explain

    @staticmethod
    def discount_rewards(r, gamma = 0.99):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    @staticmethod
    def get_baseline(r, baseline_value = 0, states = None, baseline_model = None):
        baseline = torch.zeros_like(r)
        if baseline_model is None:
            baseline += baseline_value
            return baseline
        else:
            states = states.to(next(baseline_model.parameters()).device)
            baseline = baseline_model(states).squeeze(-1)
        return baseline

    # -------------------------------------------------------------- #
    # 1.  PICK ACTION                                                #
    # -------------------------------------------------------------- #
    def get_action(self, states, evaluation=False):
        x = torch.from_numpy(np.asarray(states)).float().to(self.train_device)
        distribution = self.policy(x)

        if evaluation:
            actions = torch.tanh(distribution.mean) * self.max_action
            return actions.detach().cpu().numpy(), None

        pre_tanh_action = distribution.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        actions = tanh_action * self.max_action

        log_probs = distribution.log_prob(pre_tanh_action)
        log_probs -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim = -1, keepdim = True)

        return actions.detach().cpu().numpy(), log_probs
    
    # -------------------------------------------------------------- #
    # 2.  STORE STEP                                                 #
    # -------------------------------------------------------------- #
    def store_outcome(self, env_idx: int, state, log_prob, reward, done: bool):
        self.states_buffer[env_idx].append(torch.from_numpy(state).float())
        self.action_log_probs_buffer[env_idx].append(log_prob)
        self.rewards_buffer[env_idx].append(torch.tensor([reward], dtype=torch.float32))

        if done:
            states = torch.stack(self.states_buffer[env_idx])
            log_probs = torch.stack(self.action_log_probs_buffer[env_idx]).squeeze(-1)
            rewards = torch.stack(self.rewards_buffer[env_idx]).squeeze(-1)

            disc_returns = self.discount_rewards(rewards, self.gamma)
            self.episode_data.append({
                "states": states,
                "log_probs": log_probs,
                "returns": disc_returns,
            })

            self.states_buffer[env_idx] = []
            self.action_log_probs_buffer[env_idx] = []
            self.rewards_buffer[env_idx] = []

    # -------------------------------------------------------------- #
    # 3.  UPDATE                                                     #
    # -------------------------------------------------------------- #
    def update_policy(self):
        if not self.episode_data:
            return 0, 0

        log_probs = torch.cat([ep["log_probs"] for ep in self.episode_data]).to(self.train_device)
        states = torch.cat([ep["states"] for ep in self.episode_data]).to(self.train_device)
        returns = torch.cat([ep["returns"] for ep in self.episode_data]).to(self.train_device)

        self.episode_data = []

        pred_baseline = self.get_baseline(returns, baseline_value=self.baseline, states=states, baseline_model=self.baseline_model)
        advantage = returns - pred_baseline.detach()
        actor_loss = -(log_probs * advantage).sum()

        if self.baseline_model is not None:
            baseline_loss = torch.nn.functional.mse_loss(pred_baseline, returns.detach())
            self.optimizer_baseline.zero_grad()
            baseline_loss.backward()
            self.optimizer_baseline.step()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        adv_std = advantage.std().item()
        return actor_loss, adv_std
