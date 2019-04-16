#!/usr/bin/env python3
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd

def doChart():
    # Make a data frame
    df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21), 'y4': np.random.randn(10)+range(6,16), 'y5': np.random.randn(10)+range(4,14)+(0,0,0,0,0,0,0,-3,-8,-6), 'y6': np.random.randn(10)+range(2,12), 'y7': np.random.randn(10)+range(5,15), 'y8': np.random.randn(10)+range(4,14), 'y9': np.random.randn(10)+range(4,14), 'y10': np.random.randn(10)+range(2,12) })
 
    # style
    plt.style.use('seaborn-darkgrid')
 
    # create a color palette
    palette = plt.get_cmap('Set1')
 
    # multiple line plot
    num=0
    for column in df.drop('x', axis=1):
        num+=1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
    # Add legend
    plt.legend(loc=2, ncol=2)
 
    # Add titles
    plt.title("Results", loc='center', fontsize=14, fontweight=0, color='darkred')
    plt.xlabel("Time")
    plt.ylabel("Score")


def doThings():
    for episode in range(args.episodes):
        env.reset()

        # TODO: Initialize required values (depending on mode).
        whatIKnow = args.initial*np.ones(args.bandits,dtype=float)
        whatIKnowC = np.ones(args.bandits, dtype = int)
        t = 1
        
        average_rewards.append(0)
        done = False
        while not done:
            res = None
            # TODO: Action selection according to mode
            if args.mode == "greedy":
                greedy = np.random.randint(0, args.epsilon - 1)
                if(greedy == 0):
                    action = np.random.randint(0, args.bandits - 1)
                else:
                    action = whatIKnow.argmax(axis = 0)
            elif args.mode == "ucb":
                forArgMax=countUCB(whatIKnow,whatIKnowC,t)
                action = forArgMax.argmax(axis = 0)
            elif args.mode == "gradient":
                res = softmax(whatIKnow)
                action = res.argmax(axis = 0)

            _, reward, done, _ = env.step(action)
            average_rewards[-1] += reward / args.episode_length

            # TODO: Update parameters
            if args.mode != "gradient" :
                whatIKnowC[action] += 1
                whatIKnow[action] = update(whatIKnow, whatIKnowC[action],action, reward)
            else :
                indicate = np.zeros(args.bandits, dtype=int)
                indicate[action] = 1
                whatIKnow = whatIKnow + args.alpha*(indicate - res)
                t += 1

    # Print out final score as mean and variance of all obtained rewards.
    meanRew = np.mean(average_rewards)
    print("Final score: {}, variance: {}".format(meanRew, np.var(average_rewards)))
    return meanRew

def update(rewards, count, action, reward):
    if args.mode == "greedy" :
        if args.alpha == 0:
            a = 1/count
        else:
            a = args.alpha
        return rewards[action] + a*(reward - rewards[action])

def countUCB(rewards,counts,t):
    res = rewards
    res2 = np.reciprocal(counts.astype(float))
    return res + args.c *  np.sqrt(mt.log(t,2)*res2)

def softmax(Ht):
    ex = np.exp(Ht)
    exSum = sum(ex)
    return ex//exSum
    
    

class MultiArmedBandits():
    def __init__(self, bandits, episode_length):
        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(np.random.normal(0., 1.)) #stredni hodntoty banditu
        self._done = True
        self._episode_length = episode_length
        #print("Initialized {}-armed bandit, maximum average reward is {}".format(bandits, np.max(self._bandits)))

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = np.random.normal(self._bandits[action], 1.) #nahodna odmena z normal dist s jejich stredni hodnotou a rozptylem 1
        return None, reward, self._done, {}


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")

    parser.add_argument("--mode", default="ucb", type=str, help="Mode to use -- greedy, ucb and gradient.")
    parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
    parser.add_argument("--c", default=1., type=float, help="Confidence level in ucb.")
    parser.add_argument("--epsilon", default= 64, type=float, help="Exploration factor (if applicable).")
    parser.add_argument("--initial", default=0, type=float, help="Initial value function levels.")
    args = parser.parse_args()

    env = MultiArmedBandits(args.bandits, args.episode_length)

    average_rewards = []
    
    args.mode = "greedy"
    #greedy
    args.alpha = 0
    args.initial = 0
    for e in [64,32, 16, 18, 4]:
        args.epsilon = e
        doThings()
        
    #greedy with alpha
    args.alpha = 0.15
    args.initial = 0
    for e in [64,32, 16, 18, 4]:
        args.epsilon = e
        doThings()
    
    #greedy with alpha and initial
    args.alpha = 0.15
    args.initial = 1
    for e in [128, 64, 32, 16]:
        args.epsilon = e
        doThings()
    
    #ucb
    args.mode = "ucb"
    for c in [1./4,1./2,1.,2.,4.]:
        args.c = c
        doThings()
    
    #gradient
    args.mode = "gradient"
    for a in [1./16,1./8,1./4,1./2]:
        args.alpha = a
        doThings()
        
        
        
        ###################################
         # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")

    parser.add_argument("--mode", default="ucb", type=str, help="Mode to use -- greedy, ucb and gradient.")
    parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
    parser.add_argument("--c", default=1., type=float, help="Confidence level in ucb.")
    parser.add_argument("--epsilon", default= 64, type=float, help="Exploration factor (if applicable).")
    parser.add_argument("--initial", default=0, type=float, help="Initial value function levels.")
    args = parser.parse_args()

    env = MultiArmedBandits(args.bandits, args.episode_length)

    results = []
    args.mode = "greedy"
    nove = []
    #greedy
    x = []
    y =[]
    args.alpha = 0
    args.initial = 0
    for e in [64,32, 16, 8, 4]:
        args.epsilon = e
        res = doThings()
        x.append(Fraction(1,e))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    print(results)
     
    #greedy with alpha
    nove = []
    x = []
    y =[]
    args.alpha = 0.15
    args.initial = 0
    for e in [64,32, 16, 8, 4]:
        args.epsilon = e
        res = doThings()
        x.append(Fraction(1,e))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    #greedy with alpha and initial
    nove = []
    x = []
    y =[]
    args.alpha = 0.15
    args.initial = 1
    for e in [128, 64, 32, 16]:
        args.epsilon = e
        res = doThings()
        x.append(Fraction(1,e))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    #ucb
    nove = []
    x = []
    y =[]
    args.mode = "ucb"
    for c in [Fraction(1,4),Fraction(1,2),1.,2.,4.]:
        args.c = c
        res = doThings()
        x.append(c)
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    #gradient
    nove = []
    x = []
    y =[]
    args.mode = "gradient"
    for a in [16,8,4,2]:
        args.alpha = np.reciprocal(float(a))
        res = doThings()
        x.append(Fraction(1,a))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    print(results)        