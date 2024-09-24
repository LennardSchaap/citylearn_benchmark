import argparse
import concurrent.futures
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
import inspect

from pprint import pprint

import gymnasium
import numpy as np
import pandas as pd
import torch as th
import wandb

# Independent training
import subprocess
import shutil
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from wandb.integration.sb3 import WandbCallback

from citylearn.agents.rbc import OptimizedRBC, BasicRBC
from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from preprocess_central import get_settings, get_timestamps

# Suppress Gymnasium warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def load_config(file_name='training_config.json'):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(file_path, 'r') as file:
        return json.load(file)

class SimulationManager:
    def __init__(self, agent_type, **kwargs):
        self.agent_type = agent_type.lower()
        self.settings = get_settings()
        self.timestamps = get_timestamps()
        self.training_config = load_config()
        self.kwargs = kwargs
        self.schema = read_json(os.path.join(self.settings['schema_directory'], kwargs['schema']))

        if self.training_config["training_type"] == "central":
            self.simulation_id = f"central_agent_{self.schema['simulation_id']}_seed_{self.training_config['seed']}"
            self.simulation_output_path = os.path.join(self.settings['simulation_output_directory'], self.simulation_id)
        else:
            top_level_dir = f"independent_agent_{self.training_config['no_buildings']}_{self.training_config['algorithm']}_{self.training_config['seed']}"            # Subfolder named based on simulation ID and seed
            self.simulation_id = kwargs.get('simulation_id', f"{self.schema['simulation_id']}_seed_{self.training_config['seed']}")            # Full path with both top-level folder and subfolder
            self.simulation_output_path = os.path.join(self.settings['simulation_output_directory'], top_level_dir, self.simulation_id)

        self.project_name = f"sb3_{self.training_config['training_type']}_{self.training_config['no_buildings']}_buildings"
        self.model_save_file_name = f"{self.project_name}_{self.training_config['algorithm']}_{self.training_config['seed']}"
        self.model_save_path = os.path.join(self.training_config["data_directory"], "models")

        self.total_timesteps = None
        self.data_saver = DataSaver(self.simulation_id, self.simulation_output_path, self.timestamps, self.training_config)

        # For independent agents:
        self.building_name = kwargs.get('building')

        os.makedirs(self.simulation_output_path, exist_ok=True)
        self.initialize_schema()
        self.set_logger()

    def initialize_schema(self):

        self.schema['root_directory'] = os.path.split(Path(self.kwargs['schema']).absolute())[0]
        self.schema['episodes'] = self.training_config['episodes']
        self.schema['actions']['dhw_storage']['active'] = self.training_config["use_dhw_storage"]
        self.schema['actions']['electrical_storage']['active'] = self.training_config["use_electrical_storage"]

        if self.agent_type == 'independent':
            # Independent agents for each building, only include the specified building
            if self.kwargs.get('building', None) is not None:
                for b in self.schema['buildings']:
                    self.schema['buildings'][b]['include'] = (b == self.kwargs['building'])
                self.schema['central_agent'] = False
            else:
                raise ValueError("Building name must be specified for independent agent training.")
        else:
            # Central agent setup
            self.schema['central_agent'] = True

    def set_logger(self):
        log_filepath = os.path.join(self.simulation_output_path, f'{self.simulation_id}.log')
        handler = logging.FileHandler(log_filepath, mode='w')
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)

    def setup_environment(self):
        env = CityLearnEnv(self.schema, central_agent=True, random_seed=self.training_config["seed"])
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3Wrapper(env)

        policy_kwargs = self.get_policy_kwargs()
        return env, policy_kwargs

    def get_policy_kwargs(self):
        if self.training_config["no_buildings"] in [5, 10]:
            return {"n_critics": 1} if self.training_config["algorithm"] not in ["PPO", "TD3"] else {}
        
        policy_kwargs = {
            "activation_fn": th.nn.ReLU,
            "net_arch": {
                "pi": [self.training_config["neurons_per_layer"]] * self.training_config["n_layers"],
                "qf": [self.training_config["neurons_per_layer"]] * self.training_config["n_layers"]
            }
        }

        if self.training_config["algorithm"] != "PPO":
            policy_kwargs["n_critics"] = 1
        if self.training_config["algorithm"] == "SAC":
            policy_kwargs["use_sde"] = False
            
        return policy_kwargs

    def load_model(self, model_class, env, policy_kwargs):
        model_path = os.path.join(self.model_save_path, f"{self.simulation_id}")
        return model_class.load(model_path, env=env, policy_kwargs=policy_kwargs, verbose=2, device=self.training_config["device"])
        
    def setup_model(self, env, policy_kwargs, callbacks):
        model_class = self.get_model_class()
        if self.training_config["load_saved_model"]:
            model = self.load_model(model_class, env, policy_kwargs)
            self.load_model_data(model, env, callbacks)
        else:
            model = model_class(self.training_config["policy_type"], env, policy_kwargs=policy_kwargs, verbose=2, device=self.training_config["device"], seed=self.training_config["seed"])
        if self.training_config["log_to_wandb"]:
            self.setup_wandb_logging(callbacks)
        return model

    def load_model_data(self, model, env, callbacks):
        """
        Loads additional data like replay buffer, VecNormalize, and training state,
        and adjusts total timesteps based on restored state.
        """
        replay_buffer_path = os.path.join(self.model_save_path, f"{self.model_save_file_name}_replay_buffer.pkl")
        vecnormalize_path = os.path.join(self.model_save_path, f"{self.model_save_file_name}_vecnormalize.pkl")
        state_path = os.path.join(self.model_save_path, f"{self.model_save_file_name}_state.pkl")

        # Load replay buffer if applicable
        if self.training_config["algorithm"] in ["PPO", "SAC", "TD3", "DDPG"] and os.path.exists(replay_buffer_path):
            with open(replay_buffer_path, "rb") as file:
                model.replay_buffer = pickle.load(file)

        # Load VecNormalize if available
        if os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)

        # Load training state and update total timesteps
        if os.path.exists(state_path):
            with open(state_path, "rb") as file:
                state = pickle.load(file)
            self.restore_training_state(state, env, callbacks)

    def restore_training_state(self, state, env, callbacks):
        """
        Restores the training state (timesteps, calls, episodes) and adjusts total timesteps.
        """

        num_timesteps = state["num_timesteps"]
        n_calls = state["n_calls"]
        episodes = state["episodes"]

        print(self.total_timesteps)
        # Adjust total timesteps to account for already completed episodes
        self.total_timesteps -= (env.unwrapped.time_steps * episodes)

        # Update callback state
        callbacks[0].num_timesteps = num_timesteps
        callbacks[0].n_calls = n_calls
        callbacks[0].episode = episodes

        print(f"Restored state: {episodes} episodes, {num_timesteps} timesteps, {n_calls} calls")
        print(f"Timesteps remaining: {self.total_timesteps}")

    def setup_wandb_logging(self, callbacks):
        """
        Initializes WandB logging.
        """
        if self.agent_type == 'independent':
            run_name = self.building_name
        else:
            run_name = f"{self.training_config['algorithm']}_{self.training_config['episodes']}_eps_seed_{self.training_config['seed']}"

        if not self.training_config["load_saved_model"]:
            run = wandb.init(
                project=self.project_name, config=self.training_config,
                name=run_name
            )
        else:
            run = wandb.init(
                id=run_id, project=self.project_name, config=self.training_config,
                name=run_name, resume="must"
            )

        wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{self.model_save_file_name}", verbose=0)
        callbacks.append(wandb_callback)

    def get_model_class(self):
        if self.training_config["algorithm"] == "PPO":
            return PPO
        elif self.training_config["algorithm"] == "SAC":
            return SAC
        elif self.training_config["algorithm"] == "TD3":
            return TD3
        elif self.training_config["algorithm"] == "DDPG":
            return DDPG

    def setup_callbacks(self, env, simulation_output_path, total_timesteps):
        save_data_callback = SaveDataCallback(self.data_saver, self.schema, env, self.simulation_id, simulation_output_path, self.timestamps, self.schema['episodes'], self.training_config, self.simulation_id, self.agent_type, self.building_name, verbose=2)
        callbacks = [save_data_callback]

        if self.training_config["log_to_wandb"]:
            run = wandb.init(project=self.project_name, config=self.training_config, name=self.training_config["algorithm"] + "_" + str(self.training_config["episodes"]) + "_eps")
            wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{self.simulation_id}", verbose=0)
            callbacks.append(wandb_callback)
        
        return callbacks

    def evaluate_model(self, env, model):
        eval_env = self.setup_eval_env()
        observations = eval_env.reset()
        start_timestamp = datetime.utcnow()

        if self.training_config["algorithm"] != "RBC":
            vec_env = model.get_env()
            obs = vec_env.reset()

            while not eval_env.done:
                actions, _ = model.predict(obs, deterministic=True)
                obs, _, _, _= vec_env.step(actions)
                eval_env.step(actions[0])
        else:
            print("Using RBC agent")
            eval_env = CityLearnEnv(self.schema, central_agent=True)
            obs = eval_env.reset()[0]
            model = OptimizedRBC(eval_env)
            while not eval_env.done:
                actions = model.predict(observations=obs, deterministic=True)
                obs, rew, _, _, _= eval_env.step(actions)

        if self.training_config["algorithm"] != "RBC":
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

        self.data_saver.save_data(env, start_timestamp, 0, 'test')

    def setup_eval_env(self):
        eval_schema = self.schema.copy()
        eval_schema['simulation_start_time_step'] = int(self.timestamps[self.timestamps['timestamp'] == self.settings['season_timestamps'][self.schema['season']]['test_start_timestamp']].iloc[0].name)
        eval_schema['simulation_end_time_step'] = int(self.timestamps[self.timestamps['timestamp'] == self.settings['season_timestamps'][self.schema['season']]['test_end_timestamp']].iloc[0].name)
        eval_env = CityLearnEnv(eval_schema, central_agent=True)
        eval_env = NormalizedObservationWrapper(eval_env)
        eval_env = StableBaselines3Wrapper(eval_env)
        return eval_env

    def simulate(self, **kwargs):
        env, policy_kwargs = self.setup_environment()

        self.total_timesteps = env.unwrapped.time_steps * self.training_config['episodes']
        callbacks = self.setup_callbacks(env, self.simulation_output_path, self.total_timesteps)

        # Model setup
        model = self.setup_model(env, policy_kwargs, callbacks)

        if self.training_config["algorithm"] != "RBC" or self.training_config["evaluate"]:
            model.learn(total_timesteps=self.total_timesteps, callback=callbacks)
        
        print("Evaluating the model...")
        self.evaluate_model(env, model)

class DataSaver:
    def __init__(self, simulation_id, simulation_output_path, timestamps, training_config):
        self.simulation_id = simulation_id
        self.simulation_output_path = simulation_output_path
        self.timestamps = timestamps
        self.training_config = training_config

    def save_data(self, env, start_timestamp, episode, mode):
        """
        Save the simulation data such as environment summary, rewards, and KPIs.
        """
        end_timestamp = datetime.utcnow()
        
        print(f"[DEBUG] Saving data for simulation ID {self.simulation_id}, Episode: {episode}, Mode: {mode}")

        # Save timer data
        timer_filepath = os.path.join(self.simulation_output_path, f'{self.simulation_id}-timer.csv')
        try:
            timer_data = pd.DataFrame([{
                'simulation_id': self.simulation_id,
                'mode': mode,
                'episode': episode,
                'start_timestamp': start_timestamp, 
                'end_timestamp': end_timestamp
            }])
            
            if os.path.isfile(timer_filepath):
                print(f"[DEBUG] Reading existing timer data from {timer_filepath}")
                existing_data = pd.read_csv(timer_filepath)
                timer_data = pd.concat([existing_data, timer_data], ignore_index=True, sort=False)
            
            timer_data.to_csv(timer_filepath, index=False)
            print(f"[DEBUG] Timer data saved to {timer_filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save timer data to {timer_filepath}: {e}")

        # Save environment summary data
        try:
            data_list = []
            if self.training_config["save_env_data_during_training"] or mode == "test":
                for i, b in enumerate(env.buildings):
                    print(f"[DEBUG] Collecting data for building {b.name}")
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
                    env_data['timestamp'] = self.timestamps['timestamp'].iloc[
                        env.unwrapped.schema['simulation_start_time_step']:
                        env.unwrapped.schema['simulation_start_time_step'] + env.time_step + 1
                    ].tolist()
                    env_data['time_step'] = env_data.index
                    env_data['mode'] = mode
                    env_data['episode'] = episode
                    env_data['building_id'] = i
                    env_data['building_name'] = b.name
                    env_data['simulation_id'] = self.simulation_id
                    data_list.append(env_data)

                env_filepath = os.path.join(self.simulation_output_path, f'{self.simulation_id}-environment.csv')

                if os.path.isfile(env_filepath):
                    print(f"[DEBUG] Reading existing environment data from {env_filepath}")
                    existing_data = pd.read_csv(env_filepath)
                    data_list = [existing_data] + data_list
                
                env_data = pd.concat(data_list, ignore_index=True, sort=False)
                env_data.to_csv(env_filepath, index=False)
                print(f"[DEBUG] Environment data saved to {env_filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save environment data: {e}")

        # Save reward data  
        reward_filepath = os.path.join(self.simulation_output_path, f'{self.simulation_id}-reward.csv')
        try:
            print(f"[DEBUG] Saving reward data to {reward_filepath}")
            reward_data = pd.DataFrame(env.rewards, columns=['reward'])
            reward_data['time_step'] = reward_data.index
            reward_data['building_name'] = None
            reward_data['mode'] = mode
            reward_data['episode'] = episode
            reward_data['simulation_id'] = self.simulation_id

            if os.path.isfile(reward_filepath):
                print(f"[DEBUG] Reading existing reward data from {reward_filepath}")
                existing_data = pd.read_csv(reward_filepath)
                reward_data = pd.concat([existing_data, reward_data], ignore_index=True, sort=False)

            reward_data.to_csv(reward_filepath, index=False)
            print(f"[DEBUG] Reward data saved to {reward_filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save reward data: {e}")

        # Save KPIs
        kpi_filepath = os.path.join(self.simulation_output_path, f'{self.simulation_id}-kpi.csv')
        try:
            print(f"[DEBUG] Saving KPI data to {kpi_filepath}")
            kpi_data = env.unwrapped.evaluate()
            kpi_data['mode'] = mode
            kpi_data['episode'] = episode
            kpi_data['simulation_id'] = self.simulation_id

            if os.path.isfile(kpi_filepath):
                print(f"[DEBUG] Reading existing KPI data from {kpi_filepath}")
                existing_data = pd.read_csv(kpi_filepath)
                kpi_data = pd.concat([existing_data, kpi_data], ignore_index=True, sort=False)

            kpi_data.to_csv(kpi_filepath, index=False)
            print(f"[DEBUG] KPI data saved to {kpi_filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save KPI data: {e}")

class SaveDataCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, data_saver, schema, env, simulation_id, simulation_output_path, timestamps, episodes, training_config, model_save_file_name, agent_type, building_name, verbose=0):
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
        self.training_config = training_config

        self.save_freq = training_config['model_save_freq'] * env.unwrapped.time_steps
        self.save_path = training_config["data_directory"] + "/models/"
        self.name_prefix = model_save_file_name
        self.save_replay_buffer = True
        self.save_vecnormalize = True

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.data_saver = data_saver

        # For independent agents:
        self.agent_type = agent_type
        self.building_name = building_name
        self.log_data_accumulator = []  

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
            self.data_saver.save_data(self.env, self.start_timestamp, self.episode, self.mode)
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

            if self.training_config["algorithm"] == "PPO":
                losses_dict = {
                    "loss": self.model.logger.name_to_value['train/loss'],
                    "policy_gradient_loss" : self.model.logger.name_to_value['train/policy_gradient_loss'],
                    "value_loss" : self.model.logger.name_to_value['train/value_loss']
                }
            elif self.training_config["algorithm"] in ["SAC", "TD3", "DDPG"]:
                losses_dict = {
                    "actor_loss": self.model.logger.name_to_value['train/actor_loss'],
                    "critic_loss": self.model.logger.name_to_value['train/critic_loss']
                }
            elif self.training_config["algorithm"] == "RPPO":
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

            # Accumulate logs if agent is independent
            if self.agent_type == 'independent':
                self.log_data_accumulator.append(log_data)
            else:
                if self.training_config["log_to_wandb"]:
                    wandb.log(log_data)

        if self.agent_type == 'central' and self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}.zip")
            self.model.save(model_path)

            # Save the current training state including the episode count
            state_path = os.path.join(self.save_path, f"{self.name_prefix}_state.pkl")
            state = {
                "num_timesteps": self.num_timesteps,
                "n_calls": self.n_calls,
                "episodes": self.episode
            }

            if wandb.run is not None and wandb.run.id is not None:
                state["run_id"] = wandb.run.id

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

    def _on_training_end(self):
        if self.agent_type == 'independent':
            # Save the accumulated log data to a JSON file
            output_file_path = os.path.join(self.simulation_output_path, f'{self.building_name}_wandb_data.json')
            print(f"Saving accumulated log data to {output_file_path}")
            with open(output_file_path, 'w') as json_file:
                json.dump(self.log_data_accumulator, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(prog='citylearn_benchmark', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # simulate central agent (simulate)
    subparser_simulate = subparsers.add_parser('simulate', help="Run simulation for central or independent agents")
    subparser_simulate.add_argument('schema', type=str, help="Path to the schema JSON file.")
    
    # Optional simulation_id argument for independent agents
    subparser_simulate.add_argument('simulation_id', nargs='?', type=str, help="Simulation ID for independent agent or algorithm for central agent.")
    
    # Optional building argument for independent agents
    subparser_simulate.add_argument('-b', '--building', nargs='?', dest='building', type=str, help="Building name for independent agent.")

    subparser_simulate.set_defaults(func=run_simulation)

    # simulate independent agents (run_work_order)
    subparser_run_work_order = subparsers.add_parser('run_work_order')
    subparser_run_work_order.add_argument('work_order_filepath', type=Path)
    subparser_run_work_order.set_defaults(func=run_work_order)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

def run_work_order(work_order_filepath, windows_system=None):
    settings = get_settings()
    work_order_filepath = Path(work_order_filepath)

    training_config = load_config() 
    virtual_environment_path = training_config["virtual_environment_path"]
    conda_environment = training_config["conda_environment"]

    if virtual_environment_path: 
        if windows_system:
            command = f'"{os.path.join(virtual_environment_path, "Scripts", "Activate.ps1")}"'
        else:
            command = f'source "{os.path.join(virtual_environment_path, "bin", "activate")}"'

    elif conda_environment:
        if windows_system:
            command = f'cmd.exe /C "conda activate {conda_environment}"'
        else:
            command = f'eval "$(conda shell.bash hook)" && conda activate {conda_environment}'
    else:
        print("No environment found")

    with open(work_order_filepath, mode='r') as f:
        args = f.read()

    args = args.strip('\n').split('\n')
    args = [f'{command} && {a}' for a in args]

    max_workers = settings.get('max_workers', None) or os.cpu_count()
    # max_workers = min(settings.get('max_workers', None) or os.cpu_count(), training_config['cpu_count'])

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run, **{'args': a, 'shell': True}) for a in args]

        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            except Exception as e:
                print(e)

def run_simulation(**kwargs):
    if 'building' in kwargs and kwargs['building'] is not None:
        sim_manager = SimulationManager("independent", **kwargs)
    else:
        sim_manager = SimulationManager("central", **kwargs)

    sim_manager.simulate()

if __name__ == '__main__':
    main()
