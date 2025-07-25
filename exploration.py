import torch
from torch import nn
from torch import distributions as torchd

import models
import networks
import tools


class Random(nn.Module):
    def __init__(self, config, act_space):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.actor["dist"] == "onehot":
            return tools.OneHotDist(
                torch.zeros(
                    self._config.num_actions, device=self._config.device
                ).repeat(self._config.envs, 1)
            )
        else:
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(
                        self._act_space.low, device=self._config.device
                    ).repeat(self._config.envs, 1),
                    torch.tensor(
                        self._act_space.high, device=self._config.device
                    ).repeat(self._config.envs, 1),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(self, config, world_model, reward):
        super(Plan2Explore, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        self._reward = reward
        self._behavior = models.ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )
        self._networks = nn.ModuleList(
            [networks.MLP(**kw) for _ in range(config.disag_models)]
        )
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw
        )

    def train(self, start, context, data):
        with tools.RequiresGrad(self._networks):
            metrics = {}
            stoch = start["stoch"]
            if self._config.dyn_discrete:
                stoch = torch.reshape(
                    stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                )
            target = {
                "embed": context["embed"],
                "stoch": stoch,
                "deter": start["deter"],
                "feat": context["feat"],
            }[self._config.disag_target]
            inputs = context["feat"]
            if self._config.disag_action_cond:
                inputs = torch.concat(
                    [inputs, torch.tensor(data["action"], device=self._config.device)],
                    -1,
                )
            metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self._networks], 0
        )
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self._config.disag_log:
            disag = torch.log(disag)
        reward = self._config.expl_intr_scale * disag
        if self._config.expl_extr_scale:
            reward += self._config.expl_extr_scale * self._reward(feat, state, action)
        return reward

    def _train_ensemble(self, inputs, targets):
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]
            targets = targets.detach()
            inputs = inputs.detach()
            preds = [head(inputs) for head in self._networks]
            likes = torch.cat(
                [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
            )
            loss = -torch.mean(likes)
        metrics = self._expl_opt(loss, self._networks.parameters())
        return metrics

class PlanBehavior(nn.Module):
    def __init__(self, config, world_model, taskbehavior):
        """
        PPO-based planner with primary policy only.

        Args:
            config (dict): Configuration dictionary with plan_behavior parameters.
            world_model: The Dreamer world model.
            taskbehavior: Reference actor for value estimation.
        """
        super(PlanBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        device = self._config.device

        self._world_model = world_model
        self._actor = taskbehavior
        self.train_every = self._config.train_every
        self.sub_batch_size = self._config.sub_batch_size
        self.num_epochs = self._config.num_epochs
        self.buffer_size = self._config.buffer_size
        self.clip_epsilon = self._config.clip_epsilon
        self.gamma = self._config.gamma
        self.lmbda = self._config.lmbda
        self.entropy_eps = self._config.entropy_eps
        self.num_cells = self._config.num_cells
        self.lr = self._config.lr
        self.seq_length = self._config.seq_length
        self.buffer_minimum = self._config.buffer_minimum
        self.meta_policy_input_dim_hgr = self._config.meta_policy_input_dim_hgr
        self.meta_policy_input_dim_lwr = self._config.meta_policy_input_dim_lwr
        
        meta_action_quant = self._config.meta_action_quant
        num_meta_action_lwr = self._config.num_meta_action_lwr

        action_spec_cat = CategoricalSpec(n=meta_action_quant, device=device)
        self.actor_prim = nn.Sequential(
            nn.Linear(self.meta_policy_input_dim_hgr, self.num_cells, device=device),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells, device=device),
            nn.Tanh(),
            nn.Linear(self.num_cells, meta_action_quant, device=device),
            nn.Unflatten(dim=-1, unflattened_size=(1, meta_action_quant)),
        )

        with torch.no_grad():
            last_linear = self.actor_prim[-2]  # this is the final nn.Linear before Unflatten
            assert isinstance(last_linear, nn.Linear), "Expected Linear layer before Unflatten"

            # Get current bias values
            biases = last_linear.bias.data.clone()

            # Get indices of sorted biases (descending)
            sorted_indices = torch.argsort(biases, descending=True)
            
            # Get 3 random indices in [2, 4]
            random_indices = torch.randperm(3) + 2  # i.e., values in [2, 3, 4] shuffled

            last_linear.bias.data[3] = biases[sorted_indices[0]]
            last_linear.bias.data[2] = biases[sorted_indices[1]]
            last_linear.bias.data[4] = biases[sorted_indices[random_indices[0]]]
            last_linear.bias.data[1] = biases[sorted_indices[random_indices[1]]]
            last_linear.bias.data[0] = biases[sorted_indices[random_indices[2]]]

            print("Biases after reassigning:", last_linear.bias.data)

        policy_module_prim = TensorDictModule(
            self.actor_prim,
            in_keys=["observation"],
            out_keys=["logits"],
        )

        class SummedCategorical(Categorical):
            def log_prob(self, value: torch.Tensor) -> torch.Tensor:
                logp_per_dim = super().log_prob(value)  # shape: [batch, 3]
                # sum across last dim -> [batch]
                return logp_per_dim.sum(dim=-1, keepdim=True)

        self.policy_module_prim = ProbabilisticActor(
            module=policy_module_prim,
            spec=action_spec_cat,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )
                        
        self.value_net = nn.Sequential(
            nn.Linear(self.meta_policy_input_dim_hgr, self.num_cells, device=device),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells, device=device),
            nn.Tanh(),
            nn.Linear(self.num_cells, 1, device=device),
        )

        self.value_module = ValueOperator(
            module=self.value_net,
            in_keys=["observation"],
            out_keys=["val_prim"],
        ).to(device)

        self.advantage_module = GAE(
            gamma=self.gamma, 
            lmbda=self.lmbda, 
            value_network=self.value_module, 
            average_gae=True,
        )

        self.advantage_module.set_keys(
            advantage='adv_prim',
            value='val_prim',
            value_target='valt_prim',
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module_prim,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            critic_coef=1.0,
            normalize_advantage=True,
            loss_critic_type="smooth_l1",
        )

        self.loss_module.set_keys(
            advantage="adv_prim",
            value="val_prim",
            value_target="valt_prim",
            action="action_prim",
            sample_log_prob="sample_log_prob_prim",
        )
        
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=4096),
        )
        self.replay_buffer_real = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size),
        )
        self.optim_v = torch.optim.Adam(self.value_module.parameters(), self.lr, eps=1e-07)
        self.optim_p = torch.optim.Adam(self.policy_module_prim.parameters(), self.lr, eps=1e-07)
        
        self.buffer_flag_b = False
        self.buffer_flag_e = False

    def _add_to_buffer(
            self, 
            mode,
            observation = None, action = None, 
            sample_log_prob = None,
            reward=None, done = None, entropy=None, implemented=None,
    ):
        """
        Staged buffer writing interface for real-world rollout data.

        This function accumulates necessary fields across multiple calls
        ('begin' → 'entropy' → 'end') and validates their presence using
        assertions. Once complete, the finalized entry is passed to the
        internal replay buffer via `_deliver_to_real_buffer`.

        Args:
            mode (str): One of 'begin', 'entropy', or 'end' to control state.
            observation (Tensor): Observation input at the beginning.
            action (Tensor): Primary action taken.
            sample_log_prob (Tensor): Log-probability of primary action.
            reward (Tensor): Reward received at end of step.
            done (bool): Whether the episode terminated.
            entropy (Tensor): Entropy value of the policy output.
            implemented (bool): Whether secondary plan was executed.
        """

        if mode == 'begin':
            assert not self.buffer_flag_b, "Buffer begin called while buffer already started."
            assert observation is not None, "Invalid observation"
            assert action is not None, "Invalid action"
            assert sample_log_prob is not None, "Invalid sample_log_prob"
            assert implemented is not None, "Invalid implemented"
            self.real_buffer_dict = {
                'observation': observation.detach().cpu().clone(),
                'action': action,
                'sample_log_prob': sample_log_prob,
                'implemented': implemented,
            }
            self.buffer_flag_b = True
            return 1

        elif mode == 'entropy':
            assert self.buffer_flag_b, "Entropy received before observations"
            assert entropy is not None, "Invalid entropy"
            self.real_buffer_dict['entropy'] = entropy.cpu()
            self.buffer_flag_e = True

        elif mode == 'end':
            assert self.buffer_flag_b and self.buffer_flag_e, "Buffer end called without full filling."
            assert reward is not None, "Invalid reward"
            assert done is not None, "Invalid done"
            self.real_buffer_dict['reward'] = reward
            self.real_buffer_dict['done'] = done
            self._deliver_to_real_buffer()
            self.real_buffer_dict.clear()
            self.buffer_flag_b = False
            self.buffer_flag_e = False

    def _deliver_to_real_buffer(self):
        """
        Constructs and adds a transition entry from the real buffer dict
        into the main replay buffer. Automatically detects whether secondary
        actions were present and includes them if so.
        """
        with torch.no_grad():
            bd = self.real_buffer_dict

            base_entry = {
                "observation": bd['observation'].squeeze(0),
                "action_prim": bd['action'].cpu().squeeze(-1),
                "sample_log_prob_prim": bd['sample_log_prob'].squeeze(-1).detach(),
                "implemented": bd['implemented'],
                "entropy": bd['entropy'].detach(),
                "next": {
                    "base_reward": bd['reward'].detach(),
                    "done": bd['done'],
                }
            }

            if bd['action_sec'] is not None:
                base_entry["action_sec"] = bd['action_sec'].squeeze(0).cpu()
                base_entry["sample_log_prob_sec"] = bd['sample_log_prob_sec'].detach()

            self.replay_buffer.add(TensorDict(base_entry))

    def process_ppo(self, filter_dict=None, empty=False):
        """
        Generalized PPO processing method for real and imaginary buffers.
        
        Args:
            filter_dict (dict): {
                'key': str,                       # e.g. 'implemented'
                'invert_key_flag': bool,         # True to exclude key==True entries
                'buffer': ReplayBuffer,          # Destination buffer
                'advantage': GAE                 # GAE module for advantage computation
                'clip_observation': bool,        # Whether to clip high-dimensional obs
            }
        """
        device = self._config.device
        key = filter_dict.get('key', None)
        invert = filter_dict.get('invert_key_flag', False)
        target_buffer = filter_dict['buffer']
        advantage = filter_dict['advantage']
        clip_obs = filter_dict.get('clip_observation', False)

        inds = torch.arange(len(self.replay_buffer))
        if key:
            mask = self.replay_buffer.storage[key]
            if invert:
                mask = ~mask
            inds = torch.where(mask)[0]
        filtered_inds = inds[inds <= (len(self.replay_buffer) - self.seq_length)]

        if len(filtered_inds) < 3:
            if empty:
                self.replay_buffer.empty()
            return 

        sequences = torch.stack([
            torch.arange(ind, ind + self.seq_length) for ind in filtered_inds
        ])
        base_r = self.replay_buffer.storage['next']['base_reward'][sequences].squeeze().sum(1)
        entropy = self.replay_buffer.storage['entropy'][sequences].squeeze().sum(1)
        total_reward = (base_r + entropy) * 0.5 * (1 / self.seq_length)

        traj = self.replay_buffer._storage[filtered_inds].clone()
        new_obs = torch.cat([
            traj["observation"][1:], traj["observation"][-1:]
        ], dim=0)
        traj.set(("next", "observation"), new_obs)
        traj = traj[:-1]
        traj.set(("next", "reward"), total_reward[:-1])

        if clip_obs:
            dim_h = self._config.meta_policy_input_dim_hgr
            dim_l = self._config.meta_policy_input_dim_lwr
            clip_dim = dim_h - dim_l
            traj['observation'] = traj['observation'][..., :-clip_dim]
            traj['next']['observation'] = traj['next']['observation'][..., :-clip_dim]

        traj = advantage(traj.to(device)).to('cpu')
        traj.pop('next')
        for i in range(len(traj)):
            target_buffer.add(traj[i])
        if empty:
            self.replay_buffer.empty()

    def process_all(self):
        self.process_ppo(
            {
                'key': None,                      
                'invert_key_flag': False,         
                'buffer': self.replay_buffer_real,          
                'advantage': self.advantage_module,                 
                'clip_observation': False,
            }
        )

    def train_policy(self, train_dict, logger):
        """
        Unified PPO training function for primary and secondary policies.

        Args:
            train_dict (dict): {
                'buffer': ReplayBuffer,              # Buffer to sample from
                'loss_module': ClipPPOLoss,          # PPO loss module
                'optim_policy': torch.optim.Optimizer,
                'optim_value': torch.optim.Optimizer,
                'tag': str,                          # Logging tag (e.g. 'r', 'r_sec')
            }
            logger: Logger object for scalar tracking.
        """
        buffer = train_dict['buffer']
        loss_module = train_dict['loss_module']
        optim_p = train_dict['optim_policy']
        optim_v = train_dict['optim_value']
        tag = train_dict.get('tag', '')

        buf_len = len(buffer)
        logger.scalar(f"buffer_size_{tag}", buf_len)

        if buf_len >= self.buffer_minimum:
            for _ in range(self.num_epochs):
                indices = torch.randint(0, buf_len, (self.sub_batch_size,))
                subdata = buffer.storage[indices].to('cuda')
                subdata['observation'] = subdata['observation'].clone().detach().requires_grad_(True)

                loss_vals = loss_module(subdata)
                loss_value_v = loss_vals["loss_critic"]
                loss_value_p = loss_vals["loss_objective"] + loss_vals["loss_entropy"]

                loss_value_p.backward()
                optim_p.step()
                optim_p.zero_grad()

                loss_value_v.backward()
                optim_v.step()
                optim_v.zero_grad()

                del loss_value_p
                del loss_value_v

            logger.scalar(f"policy_loss_{tag}", loss_vals["loss_objective"].detach().cpu().item())
            logger.scalar(f"value_loss_{tag}", loss_vals["loss_critic"].detach().cpu().item())
            logger.scalar(f"entropy_loss_{tag}", loss_vals["loss_entropy"].detach().cpu().item())

    def _train(self, logger):
        self.train_policy({
                'buffer': self.replay_buffer_real,              # Buffer to sample from
                'loss_module': self.loss_module,          # PPO loss module
                'optim_policy': self.optim_p,
                'optim_value': self.optim_v,
                'tag': '',                          # Logging tag (e.g. 'r', 'r_sec')
            },
            logger=logger)

    def _flow(self, start, horizon, r_weight, inp_action = None, num_choices = 256):
        """
        Roll out imagined trajectories from a starting latent state using the policy
        or provided actions. Returns computed features, combined rewards, entropies,
        and optionally prepares for flow-based selection.

        Args:
            start (dict): Initial latent state.
            horizon (int): Number of steps to roll out.
            r_weight (float): Reward multiplier (1 - r_weight used for entropy).
            inp_actions (list[Tensor], optional): Predefined action sequence.
            num_choices (int): Number of trajectories to duplicate for flow planning.

        Returns:
            actions (Tensor): Tensor of actions taken.
            None: Placeholder for compatibility.
            traj_ent (Tensor): Final latent states (for flow ranking).
            ent_mean (float): Mean entropy over rollout.
            rew_mean (float): Mean reward over rollout.
            final_feat (Tensor): Final features from last timestep.
        """
        with torch.no_grad():
            flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
            start = {k: flatten(v) for k, v in start.items()}
            start = {k: v.unsqueeze(0).repeat_interleave(num_choices, dim=0) for k, v in start.items()}
            
            entropies, rewards, feats, actions = [], [], [], []
            state = start

            for t in range(horizon):
                # Extract latent features from current state
                feat = self._world_model.dynamics.get_feat(state)

                # Use policy or predefined actions
                action = self._actor.actor(feat).rsample() if inp_action is None else inp_action[t]

                # Step the world model forward
                succ = self._world_model.dynamics.img_step(state, action)

                # Collect outputs
                entropies.append(self._world_model.dynamics.get_dist(succ).entropy())
                rewards.append(self._world_model.heads["reward"](self._world_model.dynamics.get_feat(succ)).mode())
                feats.append(feat)
                actions.append(action)

                # Update state
                state = succ

            # Stack time series results
            entropies = torch.stack(entropies).squeeze()
            rewards = torch.stack(rewards).squeeze()
            feats = torch.stack(feats)

            dopamine = (1 - r_weight) * entropies + r_weight * rewards

            del start

            return torch.stack(actions), dopamine.cpu(), entropies.mean(), rewards.mean(), feats[-1]
