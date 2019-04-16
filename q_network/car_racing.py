#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf
import random as random


import car_racing_evaluator

def selectAction(q_values):
    rand = np.random.rand()
    if(rand <= epsilon):
        return np.random.randint(0,2)
    else:
        return np.argmax(q_values, axis = 0)
    
def getBatch(inputs, batchsize, start):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    excerpt = indices[0:batchsize]
        
    return excerpt

def getArrays(buffer, indices):
#    actions=  []
#    states = []
#    q = []
#    for i in indices:
#        state,action,reward,_,nextstate  = buffer[i]
#        actions.append(action)
#        states.append(state)
#        q.append(reward + args.gamma * np.max(network.predict([nextstate])[0]))
#    return actions,states,q

    actions=  []
    states = []
    q = []
    for i in indices:
        state,action,reward,done,nextstate  = i
        actions.append(action)
        states.append(state)
        if(done):
            q.append(reward)
        else:
            q.append(reward + args.gamma * np.max(network.predict([nextstate])[0]))
    return actions,states,q
    
class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None])
            self.q_values = tf.placeholder(tf.float32, [None])

            # Compute the q_values
            flattened_images = tf.layers.flatten(self.states, name="flatten")
            hidden = flattened_images
            for _ in range(args.hidden_layers):
                hidden = tf.layers.dense(hidden, args.hidden_layer_size, activation=tf.nn.relu)
            #hidden = tf.layers.dropout(hidden,rate=0.5,training=self.training)
            self.predicted_values = tf.layers.dense(hidden, num_actions)

            # Training
            onehot = tf.one_hot(self.actions, num_actions)
            loss = tf.losses.mean_squared_error(self.q_values, tf.boolean_mask(self.predicted_values, onehot))
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def copy_variables_from(self, other):
        for variable, other_variable in zip(self.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                                            other.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
            variable.load(other_variable.eval(other.session), self.session)

    def predict(self, states):
        return self.session.run(self.predicted_values, {self.states: states})

    def train(self, states, actions, q_values):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.q_values: q_values})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=1, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=1, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

    args = parser.parse_args()

    # Create the environment
    env = car_racing_evaluator.environment()

    # TODO: Implement a variation to Deep Q Network algorithm.
    evaluating = False
    
    actionTable = [
            [-1,1,0],[0,1,0],[1,1,0],
            [-1,0,1],[0,0,1],[1,0,1],
            [-1,0,0],[0,0,0],[1,0,0]
            ]
    
     # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, 9)
    
    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
    
    epsilon = args.epsilon

    while True:
    # Perform a training episode
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and (env.episode + 1) % args.render_each == 0:
                env.render()
                
            q_values = network.predict([state])[0]
            if not evaluating:
                action = selectAction(q_values)
            else:
                action = np.argmax(network.predict([state])[0])
            
             # Append state, action, reward, done and next_state to replay_buffer
           
            actionC = actionTable[action]
            next_state, reward, done, _ = env.step(actionC)
            replay_buffer.append(Transition(state, action, reward, done, next_state))
            
           
            
            if(not evaluating and len(replay_buffer) % args.batch_size == 0 ):
                    batch = random.sample(replay_buffer,args.batch_size)
                    #batch = getBatch(replay_buffer,args.batch_size, bat * args.batch_size)
                    a,s,q = getArrays(replay_buffer,batch)
    #                i = len(replay_buffer)
    #                index = np.random.randint(0, i)
    #                chosen = replay_buffer[index]
    #                q_value = chosen.reward + args.gamma * np.max(network.predict([chosen.next_state])[0])
                    network.train(s,a,q)
                
    
            state = next_state
            
        if env.episode >= 100:
                avg = np.average(env._episode_returns[-100:])
                print(avg)
                if np.average(avg) > 400:
                    evaluating =True
    
        if not evaluating:
                if args.epsilon_final:
                    epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            
    
            #action = [0, 1, 0]