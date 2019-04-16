#!/usr/bin/env python3
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd

def doChart(results):
    # Make a data frame
    results = [[[0, 0, 0, 0, 0], [1.202179402571835, 1.341384809418804, 1.3913354383878356, 1.3865026342217504, 1.1687164283239955]], [[64, 32, 16, 18, 4], [1.3959060881559144, 1.4055527468298163, 1.4021017106870213, 1.4123116558965347, 1.1587101435713998]], [[128, 64, 32, 16], [1.5140543202505397, 1.5042261600445912, 1.4855475445681565, 1.4526802352414703]], [[4.0, 2.0, 1.0, 0.5, 0.25], [0.49697203419441127, 0.4960141410823128, 0.4959973021773181, 0.4959095157227257, 0.4959838784244496]], [[16.0, 8.0, 4.0, 2.0], [0.4966643423535347, 0.49644027121906753, 0.4974030485627046, 0.4976579700391108]]]
    # style
    plt.style.use('classic')
    plt.figure(facecolor="white")
 
    # create a color palette
    palette = plt.get_cmap('Set1')
 
    # multiple line plot
    num=0
    #for column in df.drop('x', axis=1):
     #   num+=1
      #  plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=3, alpha=1, label=column)
    for i in range(len(results.T)):
        plt.plot(results.T[i], color = palette(num))
        num += 1
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=10)
 
    # Add titles
    plt.title("Results", loc='center', fontsize=14, fontweight=0, color='black')
    plt.ylabel("Average reward",color='black')


def doThings():
    average_rewards = []
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
        x.append(np.reciprocal(e))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
     
    #greedy with alpha
    nove = []
    x = []
    y =[]
    args.alpha = 0.15
    args.initial = 0
    for e in [64,32, 16, 8, 4]:
        args.epsilon = e
        res = doThings()
        x.append(e)
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
        x.append(e)
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    #ucb
    nove = []
    x = []
    y =[]
    args.mode = "ucb"
    for c in [1./4,1./2,1.,2.,4.]:
        args.c = c
        res = doThings()
        x.append(np.reciprocal(c))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    #gradient
    nove = []
    x = []
    y =[]
    args.mode = "gradient"
    for a in [1./16,1./8,1./4,1./2]:
        args.alpha = a
        res = doThings()
        x.append(np.reciprocal(a))
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
    
    print(results)        
    doChart(results)
   
        

    
    
    