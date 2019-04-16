#!/usr/bin/env python3
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd
from fractions import Fraction

def doChart(results):
    # Make a data frame

    plt.style.use('classic')
    plt.figure(facecolor="white")
        # create a color palette
    #results = [[['1/64', '1/32', '1/16', '1/8', '1/4'], [None,1.202179402571835, 1.341384809418804, 1.3913354383878356, 1.3573656829732024, 1.1685441202263724, None, None, None, None]], [['1/64', '1/32', '1/16', '1/8', '1/4'], [None,1.3942670643932806, 1.4097935183934824, 1.4051284156665138, 1.3520189579084085, 1.1597407299727704,  None, None, None, None]], [['1/128', '1/64', '1/32', '1/16'], [1.5111044102236884, 1.5060417012130396, 1.4875024126057848, 1.4531187309974394, None,None, None,None,None, None]], [['1/4', '1/2', '1','2','4'], [None,None,None,None,None,1.441065421881735, 1.5199785954245104, 1.5131838819899621, 1.397620046087846, 1.1462550058552963]], [['1/16', '1/8', '1/4','1/2'], [None,None,None,1.3519056005823231, 1.4464883429971844, 1.491400976325972, 1.4652306902400718, None,None,None]]]
    palette = plt.get_cmap('Set1')
 
    labels = ['1/128','1/64', '1/32', '1/16', '1/8', '1/4', '1/2','1','2','4']
    legendLabels = ['greedy','greedy(alpha)', 'greedy (optimistic)','ucb','gradient']
    plt.xticks(range(len(labels)), labels)
    # multiple line plot
    num=0
    #for column in df.drop('x', axis=1):
     #   num+=1
      #  plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=3, alpha=1, label=column)
    for i in range(len(results)):
        #labels = results[i][0]
        #plt.xticks(range(len(results[i][1])), labels)
        plt.plot(results[i][1], color = palette(num), linewidth=3, alpha=1, label=legendLabels[i])
        num += 1
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=5)
 
    # Add titles
    plt.title("Results", loc='center', fontsize=14, fontweight=0, color='black')
    plt.ylabel("Average reward",color='black')
    plt.savefig("result.png")


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
                t += 1
            elif args.mode == "gradient":
                res = softmax(whatIKnow)
                action = np.random.choice(args.bandits,1, p=res)[0]

            _, reward, done, _ = env.step(action)
            average_rewards[-1] += reward / args.episode_length

            # TODO: Update parameters
            if args.mode != "gradient" :
                whatIKnowC[action] += 1
                whatIKnow[action] = update(whatIKnow, whatIKnowC[action],action, reward)
            else :
                indicate = np.zeros(args.bandits, dtype=int)
                indicate[action] = 1
                whatIKnow = whatIKnow + args.alpha*reward*(indicate - res)
       

    # Print out final score as mean and variance of all obtained rewards.
    meanRew = np.mean(average_rewards)
    print("Final score: {}, variance: {}".format(meanRew, np.var(average_rewards)))
    return meanRew

def update(rewards, count, action, reward):
    if args.mode == "greedy" :
        if args.alpha == 0:
            a = 1./count
        else:
            a = args.alpha
        return rewards[action] + a*(reward - rewards[action])
    elif args.mode == "ucb" :
        return rewards[action] + (1./count)*(reward - rewards[action])

def countUCB(rewards,counts,t):
    res = rewards
    res2 = np.reciprocal(counts.astype(float))
#    print(res)
#    print(args.c)
    return res + args.c * np.sqrt(mt.log(t,2)*res2)

def softmax(Ht):
    ex = np.exp(Ht)
    exSum = sum(ex)
    return ex/float(exSum)
    
    

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
# 
    args.mode = "greedy"
    nove = []
    #greedy
    x = ['1/64', '1/32', '1/16', '1/8', '1/4']
    y =[None]
    args.alpha = 0
    args.initial = 0
    for e in [64,32, 16, 8, 4]:
        args.epsilon = e
        res = doThings()
        y.append(res)
    y.extend([None,None,None,None])
    nove.append(x)
    nove.append(y)
    results.append(nove)
 
     
   #greedy with alpha
    args.mode = "greedy"
    nove = []
    x = ['1/64', '1/32', '1/16', '1/8', '1/4']
    y =[None]
    args.alpha = 0.15
    args.initial = 0
    for e in [64,32, 16, 8, 4]:
        args.epsilon = e
        res = doThings()
        y.append(res)
    y.extend([None,None,None,None])
    nove.append(x)
    nove.append(y)
    results.append(nove)
     
     #greedy with alpha and initial
    nove = []
    x = ['1/128','1/64', '1/32', '1/16']
    y =[]
    args.alpha = 0.15
    args.initial = 1
    for e in [128, 64, 32, 16]:
        args.epsilon = e
        res = doThings()
        y.append(res)
    y.extend([None,None,None,None,None,None])
    nove.append(x)
    nove.append(y)
    results.append(nove)
#     
     #ucb
    nove = []
    x = ['1/4', '1/2', '1', '2', '4']
    y =[None,None,None,None,None]
    args.initial = 0
    args.alpha = 0
    args.mode = "ucb"
    for c in [1./4,1./2,1.,2.,4.]:
        args.c = c
        res = doThings()
        x.append(c)
        y.append(res)
    nove.append(x)
    nove.append(y)
    results.append(nove)
     
    #gradient
    nove = []
    x = ['1/16', '1/8', '1/4','1/2']
    y =[None,None,None]
    args.mode = "gradient"
    args.initial = 0
    for a in [16,8,4,2]:
        args.alpha = np.reciprocal(float(a))
        res = doThings()
        x.append(Fraction(1,a))
        y.append(res)
    y.extend([None,None,None])
    nove.append(x)
    nove.append(y)
    results.append(nove)
     
    print(results)        
    doChart(results)
   
        

    
    
    