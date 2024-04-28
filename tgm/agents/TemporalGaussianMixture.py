import random

from tgm.agents.actions.ActionSelectionFactory import ActionSelectionFactory
from tgm.agents.datasets.Dataset import Dataset
from tgm.agents.datasets.GaussianMixtureView import GaussianMixtureView
from tgm.agents.datasets.TemporalModelView import TemporalModelView
from tgm.agents.models.GaussianMixtureModel import GaussianMixtureModel
from tgm.agents.models.TemporalModel import TemporalModel


class TemporalGaussianMixture:

    def __init__(
        self, action_selection="softmax", n_states=10, n_observations=2, n_actions=5, learning_interval=100
    ):

        # Store the frequency at which learning is performed, and the number of actions.
        self.learning_interval = learning_interval
        self.n_actions = n_actions

        # Create the perception and temporal models, as well as the action selection strategy.
        self.gm = GaussianMixtureModel(n_states, n_observations)
        self.tm = TemporalModel(n_states)
        self.action_selection = ActionSelectionFactory.create(action_selection)  # TODO should this become the planner?

        # Create the dataset and the model's views.
        self.dataset = Dataset()
        self.gm_data = GaussianMixtureView(self.dataset)
        self.tm_data = TemporalModelView(self.dataset)

    def train(self, env, debugger=None):

        # Retrieve the initial observation from the environment.
        obs = env.reset()
        self.dataset.start_new_trial(obs)

        # Train the agent.
        steps_done = 0
        while steps_done <= 300:

            # Select the next action, and execute it in the environment.
            action = self.step(obs)
            obs, reward, done, info = env.step(action)
            self.dataset.append(obs, action, reward, done)

            # If required, perform one iteration of training.
            if steps_done > 0 and steps_done % self.learning_interval == 0:
                self.learn(debugger)
                if debugger is not None:
                    debugger.after_fit()

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()
                self.dataset.start_new_trial(obs)

            # Increase the number of steps done.
            steps_done += 1

        # Close the environment.
        env.close()

    def step(self, obs):
        # TODO
        return random.randint(0, self.n_actions - 1)

    def learn(self, debugger, force_initialize=False):

        # The first time the function is called, initialize the Gaussian mixture.
        if force_initialize or not self.gm.is_initialized():
            x = self.gm_data.get()
            self.gm.initialize(x)
            if debugger is not None:
                debugger.after_initialize()

        # Update the set of data points that can be discarded.
        self.dataset.update_forgettable_set(self.gm)

        # Fit the Gaussian mixture and temporal model.
        gm_x_forget, gm_x_keep = self.gm_data.get(split=True)
        self.gm.fit(gm_x_forget, gm_x_keep, debugger=debugger)
        tm_x_forget, tm_x_keep = self.tm_data.get(self.gm, split=True)
        self.tm.fit(tm_x_forget, tm_x_keep)

        # Forget the datapoints that can be discarded.
        self.dataset.forget()

        # Update the prior parameters.
        self.gm.update_prior_parameters()
        self.tm.update_prior_parameters()
