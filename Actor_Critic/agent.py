import numpy as np
import torch
import torch.nn.functional as F
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

class Critic(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden=64):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.tanh = torch.nn.Tanh()

        self.fc1_critic = torch.nn.Linear(state_space + action_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_mean = torch.nn.Linear(self.hidden, 1)
        self.init_weights()

    def forward(self, state, action=None):
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        x = self.tanh(self.fc1_critic(x))
        x = self.tanh(self.fc2_critic(x))
        return self.fc3_critic_mean(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(m.bias)

class Agent(object):
    def __init__(self, policy: Policy, max_action, critic: Critic, device: str = 'cpu',
        gamma: float = 0.99, # tuned
        lr_policy: float = 5e-4,  # tuned
        lr_critic: float = 5e-4,
        AC_critic: str = 'Q'
    ):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.max_action = torch.tensor(max_action, device=self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_critic = (torch.optim.Adam(self.critic.parameters(), lr=lr_critic))
        self.AC_critic = AC_critic
        
        self.gamma = gamma
        self.states_buffer = []
        self.next_states_buffer = []
        self.action_log_probs_buffer = []
        self.rewards_buffer = []
        self.done_buffer = []

    # -------------------------------------------------------------- #
    # 1.  PICK ACTION                                                #
    # -------------------------------------------------------------- #
    def get_action(self, states, evaluation=False):
        x = torch.from_numpy(np.asarray(states)).float().to(self.train_device)
        dist = self.policy(x)

        if evaluation:
            actions = torch.tanh(dist.mean) * self.max_action
            return actions.detach().cpu().numpy(), None

        pre_tanh_action = dist.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        actions = tanh_action * self.max_action

        log_probs = dist.log_prob(pre_tanh_action)
        log_probs -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim = -1, keepdim = True)
        return actions.detach().cpu().numpy(), log_probs

    # -------------------------------------------------------------- #
    # 2.  STORE STEP                                                 #
    # -------------------------------------------------------------- #
    def store_outcome(self, state, next_state, log_prob, reward, done):
        self.states_buffer.append(torch.from_numpy(state).float())
        self.next_states_buffer.append(torch.from_numpy(next_state).float())
        self.action_log_probs_buffer.append(log_prob)
        self.rewards_buffer.append(torch.tensor([reward], dtype=torch.float32))
        self.done_buffer.append(done)

    # -------------------------------------------------------------- #
    # 3.  UPDATE                                                     #
    # -------------------------------------------------------------- #
    def update_policy(self):
        log_probs = (
            torch.stack(self.action_log_probs_buffer)
            .to(self.train_device)
            .squeeze(-1)
        )
        states = torch.stack(self.states_buffer).to(self.train_device)
        next_states = torch.stack(self.next_states_buffer).to(self.train_device)
        rewards = torch.stack(self.rewards_buffer).to(self.train_device).squeeze(-1)
        done = torch.tensor(self.done_buffer, dtype=torch.float32, device=self.train_device)

        self.states_buffer, self.next_states_buffer = [], []
        self.action_log_probs_buffer, self.rewards_buffer, self.done_buffer = [], [], []

        with torch.no_grad():
            r_col = rewards.unsqueeze(-1)
            done_col = done.unsqueeze(-1)
            next_act = self.policy(next_states).mean if self.AC_critic == 'Q' else None
            target = r_col + self.gamma * \
                       self.critic(next_states, next_act) * (1 - done_col)

        current_act = self.policy(states).mean.detach() if self.AC_critic == 'Q' else None
        critic_vals = self.critic(states, current_act)

        critic_loss = F.mse_loss(critic_vals, target)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # --- 2) Actor update --------------
        advantages = (target - critic_vals).detach().squeeze(-1)
        actor_loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        return actor_loss + critic_loss, advantages.cpu().numpy()