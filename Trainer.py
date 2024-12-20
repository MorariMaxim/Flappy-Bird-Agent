import torch
import torch.nn.functional as F
import numpy as np
from NN import FlappyBirdNN
from ReplayBuffer import ReplayBuffer
import flappy_bird_gymnasium
import gymnasium
import cv2
import os
import pickle
import time
from Agent import * 

class Trainer:
    def __init__(self, agent  ,frame_stack_len =4,  batch_size=32, target_update_freq=100, load = False, skip_frames = False, minimum_replay_buffer_size=10000, kaggle = False, checkpoint_period = 25000, save_replay_buffer=False):   
    
        self.skip_frames = skip_frames
        self.agent = agent
        self.replay_buffer = ReplayBuffer()
        self.minimum_replay_buffer_size = minimum_replay_buffer_size
        self.kaggle = kaggle
                        
        self.batch_size = batch_size
        self.frame_stack_len = frame_stack_len
        self.target_update_freq = target_update_freq
        self.checkpoint_period = checkpoint_period
        self.steps = 0
        self.episode = 0
        self.device = agent.device
        self.env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
        self.save_replay_buffer = save_replay_buffer
        self.best_score = 0
        
        if load:
            self.load_checkpoint(filename="./checkpoint")

    def step(self, action):
        _, reward, done, _,_ = self.env.step(action)                
        return self.get_screen(), reward, done
    
    def get_screen(self):
        return Trainer.preprocess( self.env.render())
    
    def preprocess(screen):
        
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) 
        processed_screen = cv2.resize(gray_screen, (84, 84)) 
        return processed_screen
        

    def train(self, ):         
        while True:
            self.episode += 1
            
            start_time = time.time()
            self.env.reset(seed=1)
            
            frame_stack = [self.get_screen() for _ in range(self.frame_stack_len)]             
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(frame_stack)

                cumulative_reward = 0
                done = False
                next_frame_stack = frame_stack.copy()
                if self.skip_frames:
                                    
                    for i in range(self.frame_stack_len):
                        next_state, reward, done = self.step(action)
                        cumulative_reward += reward
                        next_frame_stack.pop(0)
                        next_frame_stack.append(next_state)
                        
                        if done:
                            break    
                else:
                    next_state, cumulative_reward, done = self.step(action) 
                    next_frame_stack.pop(0)
                    next_frame_stack.append(next_state)

                self.replay_buffer.add(frame_stack, action, cumulative_reward, next_frame_stack, done)
                frame_stack = next_frame_stack

                if len(self.replay_buffer) < self.minimum_replay_buffer_size: 
                    continue

                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.train_step(batch)

                total_reward += cumulative_reward 

                if self.steps % self.target_update_freq == 0:
                    self.agent.update_target_network()
                
                if self.steps % self.checkpoint_period == 0:
                    print("Evaluating agent...")
                    reward = self.agent.evaluate()[0]
                    print(f"Reward: {reward}")
                    if reward > self.best_score:       
                        print(f"New best score: {reward}")
                        print("Saving checkpoint...")
                        self.best_score = reward
                        self.save_checkpoint(f"./checkpoint {reward}")                                                            
                    else:
                        print(f"Best score: {self.best_score}")                        
                    
                self.agent.update_epsilon()
                    

            elapsed_time = time.time() - start_time 
            
            print(f"Episode {self.episode + 1}, Steps {self.steps}, Epsilon: {self.agent.epsilon} Total Reward: {total_reward}, Time Elapsed: {elapsed_time:.2f} seconds")
 
                
 
    def train_step(self, batch, discount_factor=0.99):
        """Performs a single training step on a minibatch."""
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.Tensor(states).to(self. device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        # Compute Q-values for current states
        q_values = self.agent.main_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.agent.target_network(next_states).max(1)[0]
            target_q_values = rewards + discount_factor * next_q_values * (1 - dones)

        # Compute loss and backpropagate
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()
    
    def save_checkpoint(self, filename="./checkpoint"):
        if self.kaggle:
            filename = "/kaggle/working/checkpoint"
        checkpoint = {
            'main_network_state_dict': self.agent.main_network.state_dict(),
            'decayed_steps': self.agent.decayed_steps,
            'epsilon_init': self.agent.epsilon_init,
            'epsilon_final': self.agent.epsilon_final,
            'episode': self.episode,
            'best_score': self.best_score
            
        }
        if self.save_replay_buffer:
            checkpoint['replay_buffer'] = self.replay_buffer
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="./checkpoint"):
        if self.kaggle:
            filename = "/kaggle/working/checkpoint"
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.agent.load_from_checkpoint(filename)
            if 'replay_buffer' in checkpoint:
                self.replay_buffer = checkpoint['replay_buffer'] 
            self.steps = checkpoint['decayed_steps'] 
            self.episode = checkpoint['episode']
            self.best_score = checkpoint['best_score']

