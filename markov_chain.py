"""The Markov Chain."""
import random


class MarkovChain:
    """The class to represent Markov chains."""

    def __init__(self, transition_matrix=None, states=None, initial_state=None):
        self.transition_matrix = transition_matrix
        self.states = states
        self.current_state = initial_state

    def attain_transition_probability(self, start_state, terminal_state):
        """
        Finds the probability of a transition from one state to another.

        :param start_state: The name of the state the chain is in at the start.
        :type start_state: str
        :param terminal_state: The name of the terminal state we want the probabilty of.
        :type terminal_state: str
        :return: The probability of the transition occurring.
        :rtype: float
        """
        start_state_index = self.states.index(start_state)
        terminal_state_index = self.states.index(terminal_state)

        return self.transition_matrix[start_state_index, terminal_state_index]

    def step(self):
        """
        Step from the current state to the next state based on the probability given by the transition matrix.

        """
        current_state_index = self.states.index(self.current_state)
        transition_value = random.random()
        transition_total = 0.0
        for state_index, transition_probability in enumerate(self.transition_matrix[current_state_index]):
            transition_total += transition_probability
            if transition_total > transition_value:
                self.current_state = self.states[state_index]
                return
        assert False, "This function has a bug if this was reached."
