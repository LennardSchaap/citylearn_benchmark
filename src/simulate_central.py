import argparse
import concurrent.futures
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
import shutil
import subprocess
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3
from stable_baselines3.ddpg import DDPG

#RBC Baseline
from citylearn.agents.rbc import OptimizedRBC, BasicRBC

#RPPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

#MADDPG
from maddpg.maddpg import MADDPG
from maddpg.buffer import MultiAgentReplayBuffer

#Frame-stack
from stable_baselines3.common.vec_env import VecFrameStack

#Custom network
import torch as th

# Stable baselines 3 callbacks
from stable_baselines3.common.callbacks import BaseCallback
import pickle

import sys
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from citylearn.utilities import read_json
from preprocess_central import get_settings, get_timestamps
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium

import warnings

# Suppress Gymnasium warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# Load json config

import json

def load_config(file_name='training_config.json'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

training_config = load_config()

def simulate(**kwargs):

    settings = get_settings()
    timestamps = get_timestamps()
    schema = kwargs['schema']
    schema = read_json(os.path.join(settings['schema_directory'], schema))
    schema['root_directory'] = os.path.split(Path(kwargs['schema']).absolute())[0]
    simulation_id = kwargs.get('simulation_id', schema['simulation_id'])

    schema['episodes'] = training_config['episodes']
    schema['actions']['dhw_storage']['active'] = training_config["use_dhw_storage"]
    schema['actions']['electrical_storage']['active'] = training_config["use_electrical_storage"]

    # set buildings
    if kwargs.get('building', None) is not None:
        for b in schema['buildings']:
            if b == kwargs['building']:
                schema['buildings'][b]['include'] = True
            else:
                schema['buildings'][b]['include'] = False
    else:
        pass
    
    # set simulation output path
    simulation_output_path = os.path.join(settings['simulation_output_directory'], simulation_id)
    
    if os.path.isdir(simulation_output_path):
        print("Directory already exists!")
    else:
        pass

    os.makedirs(simulation_output_path, exist_ok=True)
    set_logger(simulation_id, simulation_output_path)

    schema['central_agent'] = True

    # set env and agents
    env = CityLearnEnv(schema, central_agent=True)
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    if training_config["no_buildings"] == 5 or training_config["no_buildings"] == 10:
        policy_kwargs = {}
        if training_config["algorithm"] != "PPO" and training_config["algorithm"] != "TD3":
            policy_kwargs["n_critics"] = 1
    else:
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(
                pi=[training_config["neurons_per_layer"]] * training_config["n_layers"],
                qf=[training_config["neurons_per_layer"]] * training_config["n_layers"]
            )
        )
        if training_config["algorithm"] != "PPO":
            policy_kwargs["n_critics"] = 1
        if training_config["algorithm"] == "SAC":
            policy_kwargs["use_sde"] = False

    # Wandb testing
    episodes = env.unwrapped.schema['episodes']
    total_timesteps=(env.unwrapped.time_steps)*episodes

    project_name = "sb3_central" + "_" + str(training_config["no_buildings"]) + "_buildings"
    if training_config["use_dhw_storage"] == False:
        project_name = "sb3_central" + "_" + training_config["buildings"] + "_no_dhw_storage"

    model_save_file_name = project_name + "_" + training_config["algorithm"]
    save_data_callback = SaveDataCallback(schema, env, simulation_id, simulation_output_path, timestamps, episodes, training_config, model_save_file_name, verbose = 2)

    callbacks = [save_data_callback]

    policy_type = training_config["policy_type"]

    if training_config["algorithm"] == "PPO":
        model_class = PPO
        if training_config["frame-stack-ppo"]:
            env = make_train_env(env)
            env = VecFrameStack(env, training_config["n_stack"])
    elif training_config["algorithm"] == "SAC":
        model_class = SAC
    elif training_config["algorithm"] == "TD3":
        model_class = TD3
    elif training_config["algorithm"] == "DDPG":
        model_class = DDPG
    elif training_config["algorithm"] == "RPPO":
        model_class = RecurrentPPO
        policy_type = "MlpLstmPolicy"

    # config = {
    #     "policy_type": policy_type,
    #     "total_timesteps": total_timesteps,
    #     "env_name": "CityLearn",
    # }

    if training_config["load_saved_model"]:
        save_path = training_config["data_directory"] + "/models/"
        model_path = os.path.join(save_path, f"{model_save_file_name}")
        replay_buffer_path = os.path.join(save_path, f"{model_save_file_name}_replay_buffer.pkl")
        vecnormalize_path = os.path.join(save_path, f"{model_save_file_name}_vecnormalize.pkl")
        state_path = os.path.join(save_path, f"{model_save_file_name}_state.pkl")

        model = model_class.load(model_path, env=env, policy_kwargs=policy_kwargs, verbose=2, device=training_config["device"])

        # Load the replay buffer if applicable
        if training_config["algorithm"] in ["PPO", "SAC", "TD3", "DDPG"] and os.path.exists(replay_buffer_path):
            with open(replay_buffer_path, "rb") as file:
                model.replay_buffer = pickle.load(file)

        # Load the VecNormalize statistics if applicable
        if os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)

        # Load the saved training state
        if os.path.exists(state_path):
            with open(state_path, "rb") as file:
                state = pickle.load(file)
            num_timesteps = state["num_timesteps"]
            n_calls = state["n_calls"]
            episodes = state["episodes"]
            run_id = state["run_id"]

            # Update the callback with the current episode count
            save_data_callback.num_timesteps = num_timesteps
            save_data_callback.n_calls = n_calls
            save_data_callback.episode = episodes
            total_timesteps = total_timesteps - env.unwrapped.time_steps * episodes
            print("Timesteps to go: ", total_timesteps)
        print("Loaded saved model.")

    if training_config["log_to_wandb"]:
        if not training_config["load_saved_model"]:
            run = wandb.init(project=project_name, config=training_config, name=training_config["algorithm"] + "_" + str(training_config["episodes"]) + "_eps" )
        else:
            run = wandb.init(id = run_id, project=project_name, config=training_config, name=training_config["algorithm"] + "_" + str(training_config["episodes"]) + "_eps", resume="must")
        wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{model_save_file_name}", verbose=0)
        callbacks.append(wandb_callback)
        if not training_config["load_saved_model"]:
            model = model_class(policy_type, env, policy_kwargs=policy_kwargs, verbose=2, device=training_config["device"])
    else:
        if not training_config["load_saved_model"]:
            if not training_config["algorithm"] == "RBC":
                model = model_class(policy_type, env, policy_kwargs=policy_kwargs, verbose=2, device=training_config["device"])
            else:
                print("Using RBC agent")

    if not training_config["algorithm"] == "RBC" or training_config["evaluate"]:
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

    print("Evaluating")

    # evaluate
    season = schema['season']
    schema['simulation_start_time_step'] = int(timestamps[
        timestamps['timestamp']==settings['season_timestamps'][season]['test_start_timestamp']
    ].iloc[0].name)
    schema['simulation_end_time_step'] = int(timestamps[
        timestamps['timestamp']==settings['season_timestamps'][season]['test_end_timestamp']
    ].iloc[0].name)

    eval_env = CityLearnEnv(schema, central_agent=True)
    eval_env = NormalizedObservationWrapper(eval_env)
    eval_env = StableBaselines3Wrapper(eval_env)
    observations = eval_env.reset()
    start_timestamp = datetime.utcnow()

    if not training_config["algorithm"] == "RBC":
        vec_env = model.get_env()
        obs = vec_env.reset()

        while not eval_env.done:
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, _, _= vec_env.step(actions)
            eval_env.step(actions[0])

    else:
        eval_env = CityLearnEnv(schema, central_agent=True)
        obs = eval_env.reset()[0]
        model = OptimizedRBC(eval_env)
        while not eval_env.done:
            actions = model.predict(observations=obs, deterministic=True)
            obs, rew, _, _, _= eval_env.step(actions)

    if not training_config["algorithm"] == "RBC":
        evaluation_env = vec_env.envs[0]
    else:
        evaluation_env = eval_env

    kpis = evaluation_env.unwrapped.evaluate().pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    print(kpis)

    rows_to_average = ['1 - load_factor', 'average_daily_peak', 'electricity_consumption', 'peak_demand', 'ramping']
    filtered_kpis = kpis.loc[kpis.index.isin(rows_to_average), ['District']]
    avg = filtered_kpis['District'].mean()
    print(f"Average score: {avg:.2f}")

    save_data(schema, evaluation_env, simulation_id, simulation_output_path, timestamps, start_timestamp, 0, 'test')

class SaveDataCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, schema, env, simulation_id, simulation_output_path, timestamps, episodes, training_config, model_save_file_name, verbose=0):
        super(SaveDataCallback, self).__init__(verbose)
        self.schema = schema
        self.env = env
        self.simulation_id = simulation_id
        self.simulation_output_path = simulation_output_path
        self.timestamps = timestamps
        self.episodes = episodes
        self.episode = 0
        self.start_timestamp = datetime.utcnow()
        self.mode = 'train'

        self.save_freq = training_config['model_save_freq'] * env.unwrapped.time_steps
        self.save_path = training_config["data_directory"] + "/models/"
        self.name_prefix = model_save_file_name
        self.save_replay_buffer = True
        self.save_vecnormalize = True

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def _init_callback(self) -> None:
        if self.save_path is None:
            raise ValueError("You must specify `save_path` for saving the model.")

    def _on_step(self) -> bool:
        # print to log
        if self.env.unwrapped.time_step % 100 == 0:
            info = f'Time step: {self.env.unwrapped.time_step}/{self.env.unwrapped.time_steps - 1}, Episode: {self.episode + 1}/{self.episodes}'
            LOGGER.debug(info)
            print(info)

        # save timer data
        if self.env.unwrapped.time_step == self.env.unwrapped.time_steps - 2:
            print("Saving data...")
            save_data(
                self.schema, 
                self.env, 
                self.simulation_id, 
                self.simulation_output_path, 
                self.timestamps, 
                self.start_timestamp, 
                self.episode, 
                self.mode
            )
            print("Logging data...")
            self.episode += 1
            self.start_timestamp = datetime.utcnow()

            rewards = self.env.unwrapped.rewards[1:]
            rewards = sum(rewards, [])

            kpi_data = self.env.unwrapped.evaluate()

            # Filter relevant KPI's: 
            electricity_consumption_district = kpi_data.loc[(kpi_data['cost_function'] == 'electricity_consumption') & (kpi_data['level'] == 'district'), 'value'].values[0]
            average_daily_peak_values = kpi_data.loc[kpi_data['cost_function'] == 'average_daily_peak', 'value']
            ramping_values = kpi_data.loc[kpi_data['cost_function'] == 'ramping', 'value']
            peak_demand_values = kpi_data.loc[kpi_data['cost_function'] == 'peak_demand', 'value']
            load_factor_value = kpi_data.loc[kpi_data['cost_function'] == '1 - load_factor', 'value']

            if training_config["algorithm"] == "PPO":
                losses_dict = {
                    "loss": self.model.logger.name_to_value['train/loss'],
                    "policy_gradient_loss" : self.model.logger.name_to_value['train/policy_gradient_loss'],
                    "value_loss" : self.model.logger.name_to_value['train/value_loss']
                }
            elif training_config["algorithm"] in ["SAC", "TD3", "DDPG"]:
                losses_dict = {
                    "actor_loss": self.model.logger.name_to_value['train/actor_loss'],
                    "critic_loss": self.model.logger.name_to_value['train/critic_loss']
                }
            elif training_config["algorithm"] == "RPPO":
                losses_dict = {
                    "policy_gradient_loss": self.model.logger.name_to_value['train/policy_gradient_loss'],
                    "value_loss": self.model.logger.name_to_value['train/value_loss']
                }

            log_data = {
                "rewards": {
                    "average_reward": sum(rewards) / len(rewards)
                },
                "losses": losses_dict,
                "kpis": {
                    "average_daily_peak": np.nanmean(average_daily_peak_values),
                    "ramping": np.nanmean(ramping_values),
                    "peak_demand": np.nanmean(peak_demand_values),
                    "load_factor": load_factor_value.values[0] if not load_factor_value.empty else None,
                    "electricity_consumption_district": electricity_consumption_district
                }
            }

            print(log_data)

            # Log rewards, losses and KPI's to Wandb:
            if training_config["log_to_wandb"]:
                wandb.log(log_data)

        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}.zip")
            self.model.save(model_path)

            # Save the current training state including the episode count
            state_path = os.path.join(self.save_path, f"{self.name_prefix}_state.pkl")
            state = {
                "num_timesteps": self.num_timesteps,
                "n_calls": self.n_calls,
                "episodes": self.episode,
                "run_id" : wandb.run.id
            }
            with open(state_path, "wb") as file:
                pickle.dump(state, file)

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer"):
                replay_buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer.pkl")
                with open(replay_buffer_path, "wb") as file:
                    pickle.dump(self.model.replay_buffer, file)

            if self.save_vecnormalize and self.training_env is not None and hasattr(self.training_env, "save"):
                vecnormalize_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize.pkl")
                self.training_env.save(vecnormalize_path)

            if self.verbose > 1:
                print(f"Saving model checkpoint to {model_path}")

        else:
            pass

        return True

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

    if training_config["save_env_data_during_training"] or mode == "test":
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
    reward_data = pd.DataFrame(env.rewards, columns=['reward'])
    reward_data['time_step'] = reward_data.index
    reward_data['building_name'] = None
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
    kpi_data = env.unwrapped.evaluate()

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

def set_logger(simulation_id, simulation_output_path):
    os.makedirs(simulation_output_path, exist_ok=True)
    log_filepath = os.path.join(simulation_output_path, f'{simulation_id}.log')

    # set logger
    handler = logging.FileHandler(log_filepath, mode='w')
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

def main():
    parser = argparse.ArgumentParser(prog='bs2023', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # simulate
    subparser_simulate = subparsers.add_parser('simulate')
    subparser_simulate.add_argument('schema', type=str)
    subparser_simulate.add_argument('algorithm', type=str)
    # subparser_simulate.add_argument('-b', '--building', dest='building', type=str)
    subparser_simulate.set_defaults(func=simulate)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())
