import time
import os
import numpy as np
from itertools import chain
import torch
from datetime import datetime
import pandas as pd

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner

import wandb

def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config, simulation_id, simulation_output_path, timestamps):
        super(EnvRunner, self).__init__(config)
        self.simulation_id = simulation_id
        self.simulation_output_path = simulation_output_path
        self.timestamps = timestamps
        self.config = config
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # if episode > 0:
                #     print(rewards, step)
                #     print(self.envs.envs[0].env.rewards)
                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

                if step == self.episode_length - 3:
                    # print(step)
                    # print(len(self.envs.envs))
                    # print(self.envs.envs[0].env.rewards)
                    save_and_log_data(self.envs.envs[0].env.schema, 
                                      self.envs.envs[0].env, 
                                      self.simulation_id, 
                                      self.simulation_output_path, 
                                      self.timestamps, 
                                      episode, 
                                      model=None, 
                                      training_config=self.config)
                    #TODO: is dit goed?
                    self.envs.reset()


            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            end = time.time()
            print(
                "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                    self.all_args.scenario_name,
                    self.algorithm_name,
                    self.experiment_name,
                    episode,
                    episodes,
                    total_num_steps,
                    self.num_env_steps,
                    int(total_num_steps / (end - start)),
                )
            )

            if self.env_name == "MPE":
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if "individual_reward" in info[agent_id].keys():
                            idv_rews.append(info[agent_id]["individual_reward"])
                    train_infos[agent_id].update({"individual_rewards": np.mean(idv_rews)})
                    train_infos[agent_id].update(
                        {
                            "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                            * self.episode_length
                        }
                    )
            self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]
        
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )



########### DATA SAVE $$$$$$$$$$$$$$$$$$$


def save_and_log_data(schema, env, simulation_id, simulation_output_path, timestamps, episode, model=None, training_config=None):

    mode = 'train'
    start_timestamp = datetime.utcnow()
    save_data(
        schema, 
        env, 
        simulation_id, 
        simulation_output_path, 
        timestamps, 
        start_timestamp, 
        episode, 
        mode
    )


    rewards = env.rewards[1:]
    rewards = sum(rewards, [])

    # TODO FIX:

    kpi_data = env.evaluate()

    # Filter relevant KPI's: 
    electricity_consumption_district = kpi_data.loc[(kpi_data['cost_function'] == 'electricity_consumption') & (kpi_data['level'] == 'district'), 'value'].values[0]
    average_daily_peak_values = kpi_data.loc[kpi_data['cost_function'] == 'average_daily_peak', 'value']
    ramping_values = kpi_data.loc[kpi_data['cost_function'] == 'ramping', 'value']
    peak_demand_values = kpi_data.loc[kpi_data['cost_function'] == 'peak_demand', 'value']
    load_factor_value = kpi_data.loc[kpi_data['cost_function'] == '1 - load_factor', 'value']


    log_data = {
        "rewards": {
            "average_reward": sum(rewards) / len(rewards)
        },
        "kpis": {
            "average_daily_peak": np.nanmean(average_daily_peak_values),
            "ramping": np.nanmean(ramping_values),
            "peak_demand": np.nanmean(peak_demand_values),
            "load_factor": load_factor_value.values[0] if not load_factor_value.empty else None,
            "electricity_consumption_district": electricity_consumption_district
        }
    }

    # Log rewards, losses and KPI's to Wandb:
    if training_config["log_to_wandb"]:
        wandb.log(log_data)


def save_data(schema, env, simulation_id, simulation_output_path, timestamps, start_timestamp, episode, mode):
    end_timestamp = datetime.utcnow()
    timer_data = pd.DataFrame([{
        'simulation_id': simulation_id,
        'mode': mode,
        'episode': episode,
        'start_timestamp': start_timestamp, 
        'end_timestamp': end_timestamp
    }])
    timer_filepath = os.path.join(simulation_output_path, f'{simulation_id}-timer.csv')

    if os.path.isfile(timer_filepath):
        existing_data = pd.read_csv(timer_filepath)
        timer_data = pd.concat([existing_data, timer_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    timer_data.to_csv(timer_filepath, index=False)
    del timer_data

    # save environment summary data
    data_list = []

    for i, b in enumerate(env.buildings):
        env_data = pd.DataFrame({
            'solar_generation': b.solar_generation,
            'non_shiftable_load_demand': b.non_shiftable_load_demand,
            'dhw_demand': b.dhw_demand,
            'heating_demand': b.heating_demand,
            'cooling_demand': b.cooling_demand,
            'energy_from_electrical_storage': b.energy_from_electrical_storage,
            'energy_from_dhw_storage': b.energy_from_dhw_storage,
            'energy_from_dhw_device': b.energy_from_dhw_device,
            'energy_from_heating_device': b.energy_from_heating_device,
            'energy_from_cooling_device': b.energy_from_cooling_device,
            'energy_to_electrical_storage': b.energy_to_electrical_storage,
            'energy_from_dhw_device_to_dhw_storage': b.energy_from_dhw_device_to_dhw_storage,
            'electrical_storage_electricity_consumption': b.electrical_storage_electricity_consumption,
            'dhw_storage_electricity_consumption': b.dhw_storage_electricity_consumption,
            'dhw_electricity_consumption': b.dhw_electricity_consumption,
            'heating_electricity_consumption': b.heating_electricity_consumption,
            'cooling_electricity_consumption': b.cooling_electricity_consumption,
            'net_electricity_consumption': b.net_electricity_consumption,
            'net_electricity_consumption_without_storage': b.net_electricity_consumption_without_storage,
            'net_electricity_consumption_without_storage_and_pv': b.net_electricity_consumption_without_storage_and_pv,
            'electrical_storage_soc': np.array(b.electrical_storage.soc)/b.electrical_storage.capacity_history[0],
            'dhw_storage_soc': np.array(b.dhw_storage.soc)/b.dhw_storage.capacity,
        })
        env_data['timestamp'] = timestamps['timestamp'].iloc[
            schema['simulation_start_time_step']:
            schema['simulation_start_time_step'] + env.time_step + 1
        ].tolist()
        env_data['time_step'] = env_data.index
        env_data['mode'] = mode
        env_data['episode'] = episode
        env_data['building_id'] = i
        env_data['building_name'] = b.name
        env_data['simulation_id'] = simulation_id
        data_list.append(env_data)
    
    env_filepath = os.path.join(simulation_output_path, f'{simulation_id}-environment.csv')

    if os.path.isfile(env_filepath):
        existing_data = pd.read_csv(env_filepath)
        data_list = [existing_data] + data_list
        del existing_data
    else:
        pass
    
    env_data = pd.concat(data_list, ignore_index=True, sort=False)
    env_data.to_csv(env_filepath, index=False)
    del data_list
    del env_data

    # save reward data
    column_names = [f"{b.name}_reward" for b in env.buildings]
    reward_data = pd.DataFrame(env.rewards, columns=column_names)

    reward_data['time_step'] = reward_data.index
    reward_data['mode'] = mode
    reward_data['episode'] = episode
    reward_data['simulation_id'] = simulation_id
    reward_filepath = os.path.join(simulation_output_path, f'{simulation_id}-reward.csv')

    if os.path.isfile(reward_filepath):
        existing_data = pd.read_csv(reward_filepath)
        reward_data = pd.concat([existing_data, reward_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    reward_data.to_csv(reward_filepath, index=False)
    del reward_data

    # save KPIs
    ## building level
    kpi_data = env.evaluate()

    kpi_data['mode'] = mode
    kpi_data['episode'] = episode
    kpi_data['simulation_id'] = simulation_id
    kpi_filepath = os.path.join(simulation_output_path, f'{simulation_id}-kpi.csv')

    if os.path.isfile(kpi_filepath):
        existing_data = pd.read_csv(kpi_filepath)
        kpi_data = pd.concat([existing_data, kpi_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    kpi_data.to_csv(kpi_filepath, index=False)
    del kpi_data
