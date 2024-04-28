from tgm.agents.actions.EpsilonGreedyActionSelection import EpsilonGreedyActionSelection
from tgm.agents.actions.SelectBestAction import SelectBestAction
from tgm.agents.actions.SelectRandomAction import SelectRandomAction
from tgm.agents.actions.SoftmaxActionSelection import SoftmaxActionSelection


class ActionSelectionFactory:

    @staticmethod
    def create(name, **kwargs):
        """
        Create an action selection strategy
        :param name: the name of the action selection to create
        :param kwargs: the keyword parameters of the action selection strategy to overwrite
        :return: the action selection strategy
        """
        action_selection_strategies = {
            "epsilon_greedy": EpsilonGreedyActionSelection,
            "softmax": SoftmaxActionSelection,
            "random": SelectRandomAction,
            "best_action": SelectBestAction,
        }
        return action_selection_strategies[name](**kwargs)
