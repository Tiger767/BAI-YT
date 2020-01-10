import gym 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from paiutils.reinforcement import (
    DQNAgent, StochasticPolicy, GreedyPolicy,
    ExponentialDecay, RingMemory, Memory, GymWrapper
)
from paiutils.reinforcement_agents import (
    A2CAgent
)
from paiutils.neural_network import (
    dense
)


def create_amodel(state_shape, action_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = dense(128)(inputs)
    x = dense(128)(x)
    outputs = dense(action_shape[0], activation='softmax',
                    batch_norm=False)(x)
    
    amodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    amodel.compile(optimizer=keras.optimizers.Adam(.003),
                   loss='mse', experimental_run_tf_function=False)
    amodel.summary()
    return amodel


def create_qmodel(state_shape, action_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = dense(64)(inputs)
    x1 = dense(64)(x)
    x2 = dense(64)(x)
    #outputs = keras.layers.Dense(action_shape[0])(x)
    outputs = DQNAgent.get_dueling_output_layer(action_shape, 
                                                dueling_type='avg')(x1, x2)
    qmodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    qmodel.compile(optimizer=keras.optimizers.Adam(.01),
                   loss='mae', experimental_run_tf_function=False)
    qmodel.summary()
    return qmodel


def create_cmodel(state_shape):
    inputs = keras.layers.Input(shape=state_shape)
    x = dense(64)(inputs)
    x = dense(64)(x)
    outputs = keras.layers.Dense(1)(x)

    cmodel = keras.Model(inputs=inputs,
                         outputs=outputs)
    cmodel.compile(optimizer=keras.optimizers.Adam(.01),
                   loss='mse', experimental_run_tf_function=False)
    cmodel.summary()
    return cmodel


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True   
    sess = tf.compat.v1.Session(config=config)

    # Solved = Undefined ~= -42.37 avg. reward over 100 episodes

    solved = -42.37
    save_dir = ''
    env = gym.make('Acrobot-v1')
    max_steps = env._max_episode_steps  # (500)
    env = GymWrapper(env, (6,), (3,))

    agents_to_use = ['DQN', 'A2C']
    agent_to_use = agents_to_use[0]

    if agent_to_use == 'DQN':
        policy = StochasticPolicy(
            GreedyPolicy(), ExponentialDecay(1, .001, .01),
            .01, env.action_shape[0]
        )
        qmodel = create_qmodel(env.state_shape, env.action_shape)
        agent = DQNAgent(policy, qmodel, .99,
                         create_memory=lambda: RingMemory(200000),
                         enable_target=True, enable_double=False, 
                         enable_PER=False)

        agent.set_playing_data(training=False, memorizing=True)
        env.play_episodes(agent, 200, max_steps, random=True,
                          verbose=True, episode_verbose=False,
                          render=False)

        agent.set_playing_data(training=True, memorizing=True, 
                               learns_in_episode=False, batch_size=32, 
                               mini_batch=10000, epochs=1,
                               verbose=True, target_update_interval=1, tau=.01)
        for ndx in range(1):
            print(f'Save Loop: {ndx}')
            env.play_episodes(agent, 1, max_steps,
                              verbose=True, episode_verbose=False,
                              render=True)
            result = env.play_episodes(agent, 19, max_steps,
                                       verbose=True, episode_verbose=False,
                                       render=False)
            agent.save(save_dir, note=f'DQN_{ndx}_{result}')
            if result >= solved:
                break

        agent.set_playing_data(training=False, memorizing=False)
        avg = env.play_episodes(agent, 100, max_steps,
                                verbose=True, episode_verbose=False,
                                render=False)
        print(len(agent.states))
        print(avg)
    elif agent_to_use == 'A2C':
        amodel = create_amodel(env.state_shape, env.action_shape)
        cmodel = create_cmodel(env.state_shape)
        agent = A2CAgent(amodel, cmodel, .99, lambda_rate=.95,
                         create_memory=lambda: RingMemory(200000))

        agent.set_playing_data(training=False, memorizing=True)
        env.play_episodes(agent, 200, max_steps, random=True,
                          verbose=True, episode_verbose=False,
                          render=False)

        agent.set_playing_data(training=True, memorizing=True,
                               batch_size=64, mini_batch=10000, epochs=1,
                               entropy_coef=0.0,
                               verbose=True)
        for ndx in range(18):
            print(f'Save Loop: {ndx}')
            env.play_episodes(agent, 1, max_steps,
                              verbose=True, episode_verbose=False,
                              render=True)
            result = env.play_episodes(agent, 19, max_steps,
                                       verbose=True, episode_verbose=False,
                                       render=False)
            agent.save(save_dir, note=f'A2C_{ndx}_{result}')
            if result >= solved:
                break

        agent.set_playing_data(training=False, memorizing=False)
        avg = env.play_episodes(agent, 100, max_steps,
                                verbose=True, episode_verbose=False,
                                render=False)
        print(len(agent.states))
        print(avg)