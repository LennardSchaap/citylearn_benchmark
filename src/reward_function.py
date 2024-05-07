from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction
from citylearn.cost_function import CostFunction

class CustomReward(RewardFunction):
    """Calculates custom user-defined multi-agent reward.

    Reward is the :py:attr:`net_electricity_consumption_emission`
    for entire district if central agent setup otherwise it is the
    :py:attr:`net_electricity_consumption_emission` each building.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
        self.env = env_metadata

    def calculate(self) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        building = self.env.buildings[0]

        # Calculate cost functions
        electricity_consumption = CostFunction.electricity_consumption(building.net_electricity_consumption)[-1] / \
                                  CostFunction.electricity_consumption(building.net_electricity_consumption_without_storage)[-1]

        ramping_cost = CostFunction.ramping(self.env.net_electricity_consumption)[-1] / \
                       CostFunction.ramping(self.env.net_electricity_consumption_without_storage)[-1]

        load_factor_cost = CostFunction.load_factor(self.env.net_electricity_consumption)[-1] / \
                           CostFunction.load_factor(self.env.net_electricity_consumption_without_storage)[-1]

        average_daily_peak_cost = CostFunction.average_daily_peak(self.env.net_electricity_consumption)[-1] / \
                                  CostFunction.average_daily_peak(self.env.net_electricity_consumption_without_storage)[-1]

        peak_demand_cost = CostFunction.peak_demand(self.env.net_electricity_consumption)[-1] / \
                           CostFunction.peak_demand(self.env.net_electricity_consumption_without_storage)[-1]

        # Create a list of costs
        costs = [electricity_consumption, ramping_cost, load_factor_cost, average_daily_peak_cost, peak_demand_cost]

        return [sum(costs)]
