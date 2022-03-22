import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import DDPGAgent


class Generator(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)) 


class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + obs_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )
        self.apply(utils.weight_init)
    
    def forward(self, obs, action, next_obs):
        return self.net(torch.cat([obs, action, next_obs], axis=-1))

# TODO: (s,a,s') or technically, just s, s^' because a will add stochasticity
# TODO: Kostrikov schema, down learning rate by 0.5 every 10^5
class VariationalCuriosityModule(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, device):
        super().__init__()

        self.discriminator = Discriminator(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        self.optimizerD = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4
        )
        # self.optimizerD = torch.optim.lr_scheduler.StepLR( 
        #    torch.optim.Adam(
        #        self.discriminator.parameters(), 
        #        lr=1e-3
        #    ),
        #    step_size=10^5,
        #    gamma=0.5
        #)

        self.generator = Generator(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        # self.optimizerG = torch.optim.lr_scheduler.StepLR( 
        #     torch.optim.Adam(
        #        self.generator.parameters(), 
        #        lr=1e-3
        #    ),
        #    step_size=10^5,
        #    gamma=0.5
        # )
        self.device = device
        self.loss = nn.BCELoss()        

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        batch_size = obs.shape[0]

        true_label = torch.full(
            (batch_size, 1), 
            1.0, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        output_real = self.discriminator(obs, action, next_obs)
        error_real = F.binary_cross_entropy(output_real, true_label, reduction='none')

        fake_label = torch.full(
            (batch_size, 1), 
            0.0, 
            dtype=torch.float, 
            device=self.device,
            requires_grad=False
        )
        next_obs_hat = self.generator(obs, action)
        output_fake = self.discriminator(obs, action, next_obs_hat.detach())
        error_fake =  F.binary_cross_entropy(output_fake, fake_label, reduction='none')

        output_gen_fake = self.discriminator(obs, action, next_obs_hat.detach())
        error_gen =  F.binary_cross_entropy(output_gen_fake, true_label)

        return error_real, error_fake, error_gen

    def update(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]
        
        batch_size = obs.shape[0]
        next_obs_hat = self.generator(obs, action)

        true_label = torch.full(
            (batch_size, 1), 
            1.0, 
            dtype=torch.float, 
            device=self.device, 
        )
        self.generator.zero_grad()
        output_fake = self.discriminator(obs, action, next_obs_hat)
        error_gen = self.loss(output_fake, true_label)
        error_gen.backward()
        self.optimizerG.step()

        self.discriminator.zero_grad()
        output_real = self.discriminator(obs, action, next_obs)
        error_real = self.loss(output_real, true_label)
        error_real.backward()

        fake_label = torch.full(
            (batch_size, 1), 
            0.0, 
            dtype=torch.float, 
            device=self.device,
        )
        output_fake = self.discriminator(obs, action, next_obs_hat.detach())
        error_fake = self.loss(output_fake, fake_label)
        error_fake.backward()
        self.optimizerD.step()

        return error_real, error_fake, error_gen
        

class GanCMAgent(DDPGAgent):
    def __init__(self, icm_scale, update_encoder, **kwargs):
        super().__init__(**kwargs)
        self.icm_scale = icm_scale
        self.update_encoder = update_encoder

        self.vcm = VariationalCuriosityModule(
            self.obs_dim, 
            self.action_dim,
            self.hidden_dim,
            self.device
        ).to(self.device)

        self.vcm.train()

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        real_loss, fake_loss, gen_loss = self.vcm.update(obs, action, next_obs)

        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['vcm_real_loss'] = real_loss.item()
            metrics['vcm_fake_loss'] = fake_loss.item()
            metrics['vcm_gen_loss'] = gen_loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        error_real, error_fake, error_gen = self.vcm(obs, action, next_obs)

        reward = error_gen #* self.icm_scale
        # reward = torch.log(reward + 1.0)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_icm(obs, action, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, next_obs,
                                                       step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
