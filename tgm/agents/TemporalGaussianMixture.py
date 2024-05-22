import random

from tgm.agents.datasets.Dataset import Dataset
from tgm.agents.datasets.GaussianMixtureView import GaussianMixtureView
from tgm.agents.datasets.PlannerView import PlannerView
from tgm.agents.datasets.TemporalModelView import TemporalModelView
from tgm.agents.models.GaussianMixtureModel import GaussianMixtureModel
from tgm.agents.models.TemporalModel import TemporalModel
from tgm.agents.planners.QLearning import QLearning


class TemporalGaussianMixture:

    def __init__(self, action_selection="softmax", n_states=10, n_observations=2, n_actions=5, learning_interval=100):

        # Store the frequency at which learning is performed, the number of actions, and the number of step done.
        self.learning_interval = learning_interval
        self.n_actions = n_actions
        self.steps_done = 0

        # Create the perception and temporal models, as well as the action selection strategy.
        self.gm = GaussianMixtureModel(n_states, n_observations)
        self.tm = TemporalModel(n_actions, n_states)
        self.planner = QLearning(n_actions, action_selection)

        # Create the dataset and the model's views.
        self.dataset = Dataset()
        self.gm_data = GaussianMixtureView(self.dataset)
        self.tm_data = TemporalModelView(self.dataset)
        self.planner_data = PlannerView(self.dataset)

    def train(self, env, debugger=None):

        # Retrieve the initial observation from the environment.
        obs = env.reset()
        self.dataset.start_new_trial(obs)

        # Train the agent.
        self.steps_done = 0
        while self.steps_done <= 5000:

            # Select the next action, and execute it in the environment.
            action = self.step(obs)
            obs, reward, done, info = env.step(action)
            self.dataset.append(obs, action, reward, done)

            # If required, perform one iteration of training.
            if self.steps_done > 0 and self.steps_done % self.learning_interval == 0:
                self.learn(debugger)

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()
                self.dataset.start_new_trial(obs)

            # Increase the number of steps done.
            self.steps_done += 1

        # Close the environment.
        env.close()

    def step(self, obs):

        # Perform inference.
        try:
            state = self.gm.compute_responsibilities(obs.unsqueeze(dim=0))
        except TypeError:
            return random.randint(0, self.n_actions - 1)

        # Perform planning and action selection.
        return self.planner.step(state, self.steps_done)

    def learn(self, debugger, force_initialize=True):

        # The first time the function is called, initialize the Gaussian mixture.
        if force_initialize or not self.gm.is_initialized():
            x = self.gm_data.get()
            self.gm.initialize(x, debugger)

        # Update the set of data points that can be discarded.
        self.dataset.update_forgettable_set(self.gm)

        # Fit the Gaussian mixture and temporal model.
        gm_x_forget, gm_x_keep = self.gm_data.get(split=True)
        debugger.before("gm_fit", auto_index=True)
        self.gm.fit(gm_x_forget, gm_x_keep, debugger)
        debugger.after("gm_fit")
        tm_x_forget, tm_x_keep = self.tm_data.get(self.gm, split=True)
        debugger.before("tm_fit", auto_index=True)
        self.tm.fit(self.gm, tm_x_forget, tm_x_keep)
        debugger.after("tm_fit")

        # Update the Q-values.
        debugger.before("planner_fit", auto_index=True)
        planner_x = self.planner_data.get(self.gm)
        self.planner.fit(planner_x, self.gm, self.tm.B())
        debugger.after("planner_fit")

        # Update the components which are considered fixed.
        self.gm.update_fixed_components(debugger)

        # Forget the datapoints that can be discarded.
        self.dataset.forget()

        # Update the prior parameters.
        self.gm.update_prior_parameters()
        self.tm.update_prior_parameters()

    def clone(self):
        dataset = self.dataset.clone()
        return {
            "gm": self.gm.clone(),
            "tm": self.tm.clone(),
            "planner": self.planner.clone(),
            "dataset": dataset,
            "gm_data": GaussianMixtureView(dataset),
            "tm_data": TemporalModelView(dataset),
            "planner_data": PlannerView(dataset)
        }
