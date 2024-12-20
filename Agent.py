import torch 
import numpy as np
from NN import FlappyBirdNN 
import flappy_bird_gymnasium
import gymnasium 
from Trainer import *



class Agent:
    def __init__(self, num_actions = 2, epsilon_init = 1, epislon_final = 0.05, epsilon_decay = 50_000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_final = epislon_final
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.decayed_steps = 0

        # Initialize main and target networks
        
        
        self.main_network = FlappyBirdNN().to(self.device)
        self.target_network = FlappyBirdNN().to(self.device)


        # Copy initial weights from main to target network
        self.update_target_network()

        # Optimizer for the main network
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=1e-4)

    def load_from_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.main_network.load_state_dict(checkpoint['main_network_state_dict'])
        self.update_target_network()
        self.decayed_steps = checkpoint['decayed_steps']
        self.epsilon_init = checkpoint['epsilon_init']
        self.epsilon_final = checkpoint['epsilon_final']
        self.episode = checkpoint['episode']

    def update_target_network(self):
        """Copies the weights from the main network to the target network."""
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def update_epsilon(self):
        if self.decayed_steps < self.epsilon_decay:
            self.epsilon = self.epsilon_init + (self.epsilon_final - self.epsilon_init) * (self.decayed_steps / self.epsilon_decay)        
        
        self.decayed_steps += 1


    def predict(self, state, target=False):
        """Predicts Q-values for a given state using the main or target network."""
        network = self.target_network if target else self.main_network
        
        # Convert the state into a single numpy array before converting to a tensor
        state = torch.Tensor(np.array(state)).to(self.device).unsqueeze(0) 
        
        with torch.no_grad():
            return network(state)


    def act(self, state, use_epsilon=True): 
        """Chooses an action using an epsilon-greedy strategy."""
        if  use_epsilon and  np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)  # Random action
        else:
            q_values = self.predict(state)
            return torch.argmax(q_values).item()    


    def evaluate(self, seed=1):                
        """Evaluates the agent's performance over a number of episodes."""
        env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False) 
        total_reward = 0
        
        env.reset(seed=seed)
        frame_stack = [Trainer.preprocess(env.render()) for _ in range(4)]
        
        done = False
        frames = [env.render()]
        
        while not done:
            action = self.act(frame_stack, use_epsilon=False)
            _, reward, done, _,_ = env.step(action)
            frame = Trainer.preprocess(env.render())
            frame_stack.pop(0)
            frame_stack.append(frame)
            total_reward += reward
            frames.append(env.render())   
            
        
        print(f"Total Reward: {total_reward}")
        return total_reward, frames
    
    
