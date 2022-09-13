import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import utils

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def make_ppo_agent(obs_shape, action_shape, args, config):
    return PPO(
        state_dim=obs_shape,
        action_dim=action_shape,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        lam=config.lamda,
        K_epochs=config.K_epochs,
        eps_clip=config.eps_clip,
        has_continuous_action_space=config.has_continuous_action_space,
        max_train_steps=config.max_training_timesteps,
        action_std_init=config.action_std,
        use_gae=args.use_gae,
        use_lr_decay=args.use_lr_decay,
    )


class ActorCritic(nn.Module):
    """MLP Actor-Critic Network for PPO."""
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # critic
        self.critic = nn.Sequential(
            utils.layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            utils.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            utils.layer_init(nn.Linear(64, 1), gain=0.01),
        )

        # actor
        self.actor = nn.Sequential(
            utils.layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            utils.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            utils.layer_init(nn.Linear(64, action_dim), gain=0.01),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def get_value(self, obs):
        return self.critic(obs)

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            logits = self.actor(state)
            dist = Categorical(logits=logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            logits = self.actor(state)
            dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """
    PPO with Generalized Advantage Estimation.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, lam, K_epochs, eps_clip,
                 has_continuous_action_space, max_train_steps, action_std_init=0.6, use_gae=False, use_lr_decay=True):
        
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.max_train_steps = max_train_steps
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_gae = use_gae
        self.use_lr_decay=use_lr_decay

        self.lr_a = lr_actor
        self.lr_c = lr_critic

        self.buffer = utils.RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic, eps=1e-5)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def append_values(self, state, next_state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            value = self.policy_old.get_value(state)
            next_state = torch.FloatTensor(next_state).to(device)
            next_value = self.policy_old.get_value(next_state)

        self.buffer.next_states.append(next_state)
        self.buffer.values.append(value)
        self.buffer.next_values.append(next_value)

    def append_next_state(self, next_state):
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(device)
        
        self.buffer.next_states.append(next_state)
        
    def update(self, total_steps):

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)        

        with torch.no_grad():  # advantages and values have no gradient
            if self.use_gae:
                # Generalized Advantage Estimation
                # deltas = self.buffer.rewards + self.gamma * (1.0 - self.buffer.dw) * self.buffer.next_values - self.buffer.values
                adv = []
                gae = 0

                values = self.policy_old.get_value(old_states)
                next_values = self.policy_old.get_value(old_next_states)

                for reward, value, next_value, dw, is_terminal in zip(reversed(self.buffer.rewards), reversed(values), reversed(next_values), reversed(self.buffer.dw), reversed(self.buffer.is_terminals)):
                    delta = reward + self.gamma * (1.0 - dw) * next_value - value
                    gae = delta + self.gamma * self.lam * gae * (1.0 - is_terminal)
                    adv.insert(0, gae)
                
                # Normalizing advantages
                adv = torch.tensor(adv, dtype=torch.float32).to(device)
                rewards = adv + values
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            else:
                # Monte Carlo estimate of returns
                rewards = []
                discounted_reward = 0

                for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)
                
                # Normalizing the rewards
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)        

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            if self.use_gae:
                advantages = adv
            else:
                advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            self.optimizer_actor.zero_grad()
            actor_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            # update critic
            critic_loss = F.mse_loss(state_values, rewards)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            self.optimizer_critic.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        # learning rate Decay
        if self.use_lr_decay:  
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

