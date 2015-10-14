"""Tests related to Markov chains."""
from mock import patch, Mock, PropertyMock

import pytest
import numpy as np

from markov_chain import MarkovChain, attain_sequence_occurrences, attain_sequence_probability_distribution


class TestMarkovChain:
    """Tests for the MarkovChain class."""

    @pytest.fixture
    def chain(self):
        """A Markov chain fixture."""
        matrix = np.array([[1 / 2, 1 / 4, 1 / 4],
                           [1 / 2, 0, 1 / 2],
                           [1 / 4, 1 / 4, 1 / 2]])
        chain = MarkovChain(transition_matrix=matrix, states=['R', 'N', 'S'], initial_state='N')
        return chain

    def test_can_create_with_transition_matrix(self):
        """Tests that a transition matrix can be passed to define the chain on creation."""
        matrix = np.array([[1 / 2, 1 / 4, 1 / 4],
                           [1 / 2, 0, 1 / 2],
                           [1 / 4, 1 / 4, 1 / 2]])

        chain = MarkovChain(transition_matrix=matrix)

        assert np.array_equal(chain.transition_matrix, matrix)

    def test_state_name_retrieval_by_index(self):
        """Tests that, given a state number, the corresponding label can be retrieved."""
        chain = MarkovChain(states=['R', 'N', 'S'])

        state = chain.states[1]

        assert state == 'N'

    def test_the_transition_probability_from_one_state_to_another_can_be_retrieved(self, chain):
        """Tests that, given two state names, the transition probability can be retrieved."""

        probability0 = chain.attain_transition_probability('R', 'N')
        probability1 = chain.attain_transition_probability('S', 'S')

        assert probability0 == 1 / 4
        assert probability1 == 1 / 2

    def test_the_initial_state_can_be_set_by_state_name(self):
        """Tests that a state can be initialized."""
        chain = MarkovChain(states=['R', 'N', 'S'], initial_state='N')

        assert chain.current_state == 'N'

    @patch('random.random', side_effect=[0.6, 0.2])
    def test_can_probabilistically_transition_to_a_state(self, mock_random):
        """Tests that the chain can transition to another state based on the probability matrix."""
        matrix = np.array([[1 / 2, 1 / 4, 1 / 4],
                           [1 / 2, 0, 1 / 2],
                           [1 / 4, 1 / 4, 1 / 2]])
        chain = MarkovChain(transition_matrix=matrix, states=['R', 'N', 'S'], initial_state='N')

        chain.step()
        state1 = chain.current_state
        chain.step()
        state2 = chain.current_state

        assert state1 == 'S'
        assert state2 == 'R'

    def test_can_attain_a_sequence_of_states(self, chain):
        """
        Test that the chain can return a sequence of states.
        This test is pretty badly built. If you come up with a good way to improve it, please do so.

        :param chain: The chain object from the fixture.
        :type chain: MarkovChain
        """

        class MockChain(MarkovChain):
            def step(self):
                pass
        mock_chain = MockChain()
        mock_chain.states, mock_chain.transition_matrix = chain.states, chain.transition_matrix
        type(mock_chain).current_state = PropertyMock(side_effect=['N', 'R', 'N', 'S'])

        sequence = mock_chain.attain_sequence(length=4)

        assert sequence == ['N', 'R', 'N', 'S']

    def test_can_attaining_sequence_calls_step(self, chain):
        """
        Test that attaining the sequence results in the step being called the appropriate number of times.

        :param chain: The chain object from the fixture.
        :type chain: MarkovChain
        """
        chain.step = Mock()

        sequence = chain.attain_sequence(length=4)

        assert len(chain.step.call_args_list) == 3

    def test_can_attain_occurrence_count_for_sequence(self):
        """
        Test that the occurrence count can be given for a sequence.

        """
        sequence = ['N', 'N', 'S', 'R', 'S', 'N']

        occurrences = attain_sequence_occurrences(sequence)

        assert occurrences == {'N': 3, 'S': 2, 'R': 1}

    def test_can_attain_probability_distribution_for_sequence(self):
        """
        Test that the probability distribution can be given for a sequence.

        """
        sequence = ['N', 'N', 'S', 'R', 'S', 'N']

        distribution = attain_sequence_probability_distribution(sequence)

        assert distribution == {'N': 1/2, 'S': 1/3, 'R': 1/6}