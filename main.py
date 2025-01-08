from Agent import *
from Trainer import *

import cv2

def create_video(frames, output_path, fps=10): 
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame) 
    out.release()
 


mode = "evaluate"  # "train" or "evaluate"

if __name__ == "__main__": 
    
    if mode == "train":
        trainer = Trainer(Agent( epsilon_init=1, epislon_final=0.05, epsilon_decay=750_000), load=False) 
        
        trainer.train()
        
    else:
        agent = Agent()
        agent.load_from_checkpoint("./checkpoint (523, epsilon_init=0.1, epislon_final=0.001, epsilon_decay=1_000_000, but actually for 1.5kk)")  
                        
        frames = agent.evaluate(seed=1)[1]
                
        create_video(frames, "animation.mp4", fps=30) 
         