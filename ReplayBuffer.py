import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000, state_shape=(84, 84), frame_stack_len=4):
        self.capacity = capacity
        self.state_shape = state_shape
        self.frame_stack_len = frame_stack_len
        self.buffer = {
            'states': np.zeros((capacity, frame_stack_len, *state_shape), dtype=np.float32),
            'actions': np.zeros((capacity,), dtype=np.int64),
            'rewards': np.zeros((capacity,), dtype=np.float32),
            'next_states': np.zeros((capacity, frame_stack_len, *state_shape), dtype=np.float32),
            'dones': np.zeros((capacity,), dtype=bool)
        }
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done): 
        self.buffer['states'][self.position] = state
        self.buffer['actions'][self.position] = action
        self.buffer['rewards'][self.position] = reward
        self.buffer['next_states'][self.position] = next_state
        self.buffer['dones'][self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size): 
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = self.buffer['states'][indices]
        actions = self.buffer['actions'][indices]
        rewards = self.buffer['rewards'][indices]
        next_states = self.buffer['next_states'][indices]
        dones = self.buffer['dones'][indices]
        return states, actions, rewards, next_states, dones

    def __len__(self): 
        return self.size