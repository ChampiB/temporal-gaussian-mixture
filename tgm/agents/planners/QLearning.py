import torch

from tgm.agents.actions.ActionSelectionFactory import ActionSelectionFactory
from tgm.agents.models.display.MatPlotLib import MatPlotLib


class QLearning:

    def __init__(self, n_actions, action_selection="softmax", initial_q_value=0, learning_rate=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.action_selection = action_selection
        self.action_strategy = ActionSelectionFactory.create(action_selection)
        self.q_values = None
        self.initial_q_value = initial_q_value
        self.fixed_gaussian = None
        self.learning_rate = learning_rate
        self.gamma = gamma

    def step(self, state, steps_done):

        # Initialize the Q-values, if required.
        if self.q_values is None:
            self.q_values = torch.ones([self.n_actions, state.shape[1]]) * self.initial_q_value

        # Compute the quality of each action in the current state.
        quality = torch.matmul(self.q_values, state[0]).unsqueeze(dim=0)

        # Select an action to execute in the environment.
        action = self.action_strategy.select(quality, steps_done)
        return action if isinstance(action, int) else action.item()

    def fit(self, planner_x, gm, B):

        # Initialize the Q-values.
        n_states = len(gm.m)
        q_values = torch.ones([self.n_actions, n_states]) * self.initial_q_value
        if self.q_values is not None and self.fixed_gaussian is not None:

            # Keep in mind the parameter of the fixed components.
            fixed_components_0 = self.fixed_gaussian.fixed_components
            v0, _, _, m0, W0 = self.fixed_gaussian.parameters_of(fixed_components_0)
            new_indices_map = self.fixed_gaussian.new_indices_of(v0, m0, W0, gm.v_hat, gm.m_hat, gm.W_hat)

            for a in range(q_values.shape[0]):
                for j in range(q_values.shape[1]):
                    if j not in new_indices_map.keys():
                        continue
                    j_new = new_indices_map[j]
                    q_values[a][j_new] = self.q_values[a][j]
        self.q_values = q_values

        # Update the Q-values.
        belief_states, actions, rewards, dones = planner_x
        for belief_state, a, reward, done in zip(belief_states, actions, rewards, dones):
            for s, state_probability in enumerate(belief_state):
                temporal_difference = reward - self.q_values[a][s]
                if done is False:
                    next_state_prob = B[a].select(dim=1, index=s)
                    temporal_difference += self.gamma * torch.inner(self.q_values.max(dim=0)[0], next_state_prob)
                self.q_values[a][s] = self.q_values[a][s] + self.learning_rate * state_probability * temporal_difference

        # Keep track of the fixed components.
        self.fixed_gaussian = gm.fixed_gaussian.clone()

    def clone(self):

        # Create a new planner.
        planner = QLearning(self.n_actions, self.action_selection, self.initial_q_value, self.learning_rate, self.gamma)

        # Clone the parameters of the planner.
        planner.q_values = None if self.q_values is None else self.q_values.clone()

        # Clone the Gaussian stability.
        planner.fixed_gaussian = None if self.fixed_gaussian is None else self.fixed_gaussian.clone()

        return planner

    def draw_q_values(self, action_names):
        if self.q_values is None:
            return MatPlotLib.draw_text(f"Matrix cannot be generated because \n q-values are none.")

        return MatPlotLib.draw_matrix(self.q_values, title=f"Q-values.", y_ticks=action_names)
