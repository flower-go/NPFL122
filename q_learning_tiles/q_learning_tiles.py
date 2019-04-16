#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator

def selectAction(state):
    rand = np.random.rand()
    if(rand <= epsilon):
        return np.random.randint(0,env.actions)
    else:
        return np.argmax(W[state].sum(axis = 0))
def selectGreedy(state):
    return np.argmax(W[state].sum(axis = 0))
    
if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.08, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles
    epsilonGreedy = True

    evaluating = False
    while not evaluating:
        # Perform a training episode
        state, done = env.reset(evaluating), False
        if(env.episode > 100 and (env.episode // 100) % 2 == 0):
            epsilonGreedy = False
        else:
            epsilonGreedy = True
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose `action` according to epsilon-greedy strategy
            if(epsilonGreedy):
                action = selectAction(state)
            else:
                action = selectGreedy(state)
            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values
            if epsilonGreedy:
                W[state,action] += alpha * (reward + args.gamma*(np.max(W[next_state].sum(axis = 0)) - W[state, action].sum(axis = 0)))

            state = next_state
            if done:
                break

        # TODO: Decide if we want to start evaluating
        if(env.episode > 100 and not epsilonGreedy):
           avg = np.average(env._episode_returns[-100:])
           #print(avg)
           if avg >= -102:
                evaluating = True

        if not evaluating and epsilonGreedy:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

    # Perform the final evaluation episodes
    while True:
        state, done = env.reset(evaluating), False
        while not done:
            # TODO: choose action as a greedy action
            action = np.argmax(W[state].sum(axis = 0))
            state, reward, done, _ = env.step(action)