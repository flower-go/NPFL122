#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

def selectAction(state):
    rand = np.random.rand()
    if(rand < args.epsilon):
        return np.random.randint(0,env.actions)
    else:
        return Q[state].argmax(axis = 0)

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=400, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.2, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # TODO: Implement Monte-Carlo RL algorithm.
    Q = (1/args.epsilon)*np.ones((env.states,env.actions))
    
    
    stateActionCount = np.zeros((env.states,env.actions))
    # The overall structure of the code follows.

    for train in range(args.episodes):
        # Perform a training episode
        state, done = env.reset(False), False       
        steps = [] #nove pro kazdou epozodu?
        train += 1
        G = 0
        
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = selectAction(state)
            next_state, reward, done, _ = env.step(action)
            steps.append([state,action,reward])
            state = next_state     
            
        for state,action,reward in reversed(steps) :
            stateActionCount[state,action]+=1
            G = args.gamma * G + reward
            Q[state][action] = Q[state][action]+ (1.0/stateActionCount[state,action])*(G - Q[state][action] )

            
    # Perform last 100 evaluation episodes  
    while True:
         state, done = env.reset(True), False
         while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = np.argmax(Q[state], axis = 0)
            next_state, reward, done, _ = env.step(action)     
            state = next_state
        
   
    