"""Tests related to Markov chains."""
from mock import patch

import pytest
import numpy as np

from markov_chain import MarkovChain


class TestMarkovChain:
    """Tests for the MarkovChain class."""

    @pytest.fixture
    def chain(self):
        """A Markov chain fixture."""
        matrix = np.array([[1/2, 1/4, 1/4],
                           [1/2,   0, 1/2],
                           [1/4, 1/4, 1/2]])
        chain = MarkovChain(transition_matrix=matrix, states=['R', 'N', 'S'])
        return chain

    def test_can_create_with_transition_matrix(self):
        """Tests that a transition matrix can be passed to define the chain on creation."""
        matrix = np.array([[1/2, 1/4, 1/4],
                           [1/2,   0, 1/2],
                           [1/4, 1/4, 1/2]])

        chain = MarkovChain(transition_matrix=matrix)

        assert np.array_equal(chain.transition_matrix, matrix)

    def test_state_name_retrieval_by_index(self):
        """Tests that, given a state number, the corresponding label can be retrieved."""
        chain = MarkovChain(states=['R', 'N', 'S'])

        state = chain.states[1]

        assert state == 'N'

    def test_the_transition_probability_from_one_state_to_another_can_be_retieved(self, chain):
        """Tests that, given two state names, the transition probabiltiy can be retrieved."""

        probability0 = chain.attain_transition_probability('R', 'N')
        probability1 = chain.attain_transition_probability('S', 'S')

        assert probability0 == 1/4
        assert probability1 == 1/2

    def test_the_intial_state_can_be_set_by_state_name(self):
        """Tests that a state can be initialized."""
        chain = MarkovChain(states=['R', 'N', 'S'], initial_state='N')

        assert chain.current_state == 'N'

    def test_can_probabilistically_transition_to_a_state(self):
        """Tests that the chain can transition to another state based on the probability matrix."""
        matrix = np.array([[1/2, 1/4, 1/4],
                           [1/2,   0, 1/2],
                           [1/4, 1/4, 1/2]])
        chain = MarkovChain(transition_matrix=matrix, states=['R', 'N', 'S'], initial_state='N')

        with patch('random.random', return_value=0.6) as mock_random:
            chain.step()
        state1 = chain.current_state

        with patch('random.random', return_value=0.2) as mock_random:
            chain.step()
        state2 = chain.current_state

        assert state1 == 'S'
        assert state2 == 'R'
