# import random
# import numpy as np
# from collections import deque

# class ReplayBuffer: 
#     def __init__(self, capacity=10000):   
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity) 

#     def add(self, val):  
#         """Adds a new experience to the buffer."""
#         self.buffer.append(val)  

#     def sample(self, batch_size): 
#         batch_size = min(batch_size, len(self.buffer))
#         """Samples a batch of experiences from the buffer."""
#         # Sample batch_size number of random indices from the deque
#         batch = random.sample(self.buffer, batch_size)  # Randomly sample from the deque
        
#         # Unzip the batch into individual components
#         frame_stack, actions, rewards, next_frame_stack, dones = map(np.stack, zip(*batch))
#         return frame_stack, actions, rewards, next_frame_stack, dones

#     def __len__(self):
#         """Returns the number of experiences currently in the buffer."""
#         return len(self.buffer)

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
        """Adds a new experience to the buffer."""
        self.buffer['states'][self.position] = state
        self.buffer['actions'][self.position] = action
        self.buffer['rewards'][self.position] = reward
        self.buffer['next_states'][self.position] = next_state
        self.buffer['dones'][self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = self.buffer['states'][indices]
        actions = self.buffer['actions'][indices]
        rewards = self.buffer['rewards'][indices]
        next_states = self.buffer['next_states'][indices]
        dones = self.buffer['dones'][indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns the number of experiences currently in the buffer."""
        return self.size