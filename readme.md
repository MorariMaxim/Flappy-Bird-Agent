# Architecture
Our NN has 3 convolutional layers, each followed by batch normalisation, and 2  linear layers. It uses ReLu as activation functions.
## layer parameters 
**1st (convolutional) layer**: in_channels=4, out_channels=32, kernel_size=8, stride=4
**2nd (convolutional) layer**: in_channels=32, out_channels=64, kernel_size=4, stride=2
**3rd (convolutional) layer**: in_channels=64, out_channels=64, kernel_size=3, stride=1
**4th (linear) layer**: in_features=64 * 7 * 7,  out_features = 512
**5th (linear) layer**: in_features=512,  out_features = 2

## other
**batch size** =  32
**optimizer**: Adam
**learning rate**: 1e-4
we used epsilon decay: starting from an initial value, it decreases monotonously over a set number of frames until a final value 


## input 
The input of the NN is a stack of 4 grayscale, 84x84 images (obtained by converting the rgb game screen to grayscale and resizing.) The images are consecutives frames of the game.

# Attempts
Across different attempts we were primarily modifying the epsilon decay values, the number of frames for how long it trained and the period of synchronizing the 2 NNs. 

## best result
We got our best score of 523 by training for 1.5kk frames with 
epsilon_init=0.1, epislon_final=0.001, epsilon_decay=1_000_000, target_update_freq = 100

## other attempts
score = 11.7 
epsilon_init=0.1, epislon_final=0.001, epsilon_decay=1_000_000, steps =1.5kk, target_update_freq = 10

score = 224 
epsilon_init=0.1, epislon_final=0.001, epsilon_decay=2_000_000; steps = 1.85 kk, target_update_freq = 100

score = ~5
epsilon_init=1, epislon_final=0.05, epsilon_decay=750_000, steps = 550 target_update_freq  = 10_000

score
epsilon_init=0.1, epislon_final=0.001, epsilon_decay=1_000_000, steps = 1kk, target_update_freq  = 10_000

Here we tried to skip frames and repeating the same action in between. I guess it didn't work.
score = 20
epsilon_init=0.1, epislon_final=0.001, epsilon_decay=1_000_000
