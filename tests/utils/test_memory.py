import unittest

import utils.memory

class TestTransition(unittest.TestCase):
    def test_init_transition(self):
        """Test initializing a transition."""
        transition = utils.memory.Transition(1, 2, 3, 4)
        self.assertIsInstance(transition, utils.memory.Transition)
        self.assertEqual(transition.state, 1)
        self.assertEqual(transition.action, 2)
        self.assertEqual(transition.next_state, 3)
        self.assertEqual(transition.reward, 4)

class TestReplayMemory(unittest.TestCase):
    def test_init_memory(self):
        """Test initializing a memory."""
        memory = utils.memory.ReplayMemory()
        self.assertIsInstance(memory, utils.memory.ReplayMemory)

    def test_push(self):
        """Test pushing a transition to the memory."""
        memory = utils.memory.ReplayMemory()
        memory.push(1, 2, 3, 4)
        self.assertEqual(len(memory), 1)

    def test_sample(self):
        """Test sampling from the memory."""
        memory = utils.memory.ReplayMemory()
        memory.push(1, 2, 3, 4)
        memory.push(5, 6, 7, 8)
        memory.push(9, 10, 11, 12)
        sample = memory.sample(2)
        self.assertEqual(len(sample), 2)
        full_memory = memory.sample(3)
        self.assertIn((1,2,3,4), full_memory)
        self.assertIsInstance(sample[0], utils.memory.Transition)

    def test_deque(self):
        """Test cyclical nature of the memory."""
        memory = utils.memory.ReplayMemory(2)
        memory.push(1, 2, 3, 4)
        memory.push(5, 6, 7, 8)
        # Since we have a max length of 2, the first transition should be removed
        memory.push(9, 10, 11, 12)
        self.assertEqual(len(memory), 2)
        self.assertIn((5,6,7,8), memory.memory)
        self.assertIn((9,10,11,12), memory.memory)
        self.assertNotIn((1,2,3,4), memory.memory)
