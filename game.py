import logging, os

import warnings
warnings.filterwarnings("ignore")

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.macl import MACLAgentTrainer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=10, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--good-mic", type=str, default=0, help="mutual information coefficient for good agents")
    parser.add_argument("--adv-mic", type=str, default=0, help="mutual information coefficient for adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--sleep-regimen", action="store_true", default=False, help="only use mic while sleeping")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(inputs=out, units=num_units, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=num_units, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=out, units=num_outputs, activation=None)
        return out

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    adv_index = 0
    player_index = 0

    action_space_n = []
    for player, action_space in env.action_spaces.items():
        action_space_n.append(action_space)

    for player, _ in env.action_spaces.items():
        if(num_adversaries > 0):
            trainer = MACLAgentTrainer("agent_%d" % player_index, model, obs_shape_n, action_space_n, player_index, arglist, agent_type="bad")
            num_adversaries -= 1
        else:
            trainer = MACLAgentTrainer("agent_%d" % player_index, model, obs_shape_n, action_space_n, player_index, arglist, agent_type="good")
        trainers.append(trainer)

        player_index += 1

    return trainers

def train(arglist):
    with U.single_threaded_session():
        if(arglist.scenario == "DouDizhu"):
            from pettingzoo.classic import dou_dizhu_v1
            env = dou_dizhu_v1.env(opponents_hand_visible=False) 
        elif(arglist.scenario == "Uno"):
            from pettingzoo.classic import uno_v1
            env = uno_v1.env(opponents_hand_visible=False)
        elif(arglist.scenario == "Texas"):
            from pettingzoo.classic import texas_holdem_no_limit_v1
            env = texas_holdem_no_limit_v1.env()
        elif(arglist.scenario == "Mahjong"):
            from pettingzoo.classic import mahjong_v1
            env = mahjong_v1.env()
        elif(arglist.scenario == "Leduc"):
            from pettingzoo.classic import leduc_holdem_v1
            env = leduc_holdem_v1.env()
        elif(arglist.scenario == "Limit"):
            from pettingzoo.classic import texas_holdem_v1
            env = texas_holdem_v1.env()
        elif(arglist.scenario == "Gin"):
            from pettingzoo.classic import gin_rummy_v1
            env = gin_rummy_v1.env(knock_reward = 0.5, gin_reward = 1.0, opponents_hand_visible = False)
        elif(arglist.scenario == "Backgammon"):
            from pettingzoo.classic import backgammon_v1
            env = backgammon_v1.env()
        else:
            print("no scenario found")
            assert(False)
        
        obs_shape_n = []
        for player, space in env.observation_spaces.items():
            val = np.product(space.shape)
            obs_shape_n.append(tuple((val,)))
        
        num_players = len(env.observation_spaces)
        print("Playing with: ", num_players, " players")

        num_adversaries = min(num_players, arglist.num_adversaries)
        # Create agent trainers
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {}'.format(arglist.good_policy))

        # Initialize
        U.initialize()

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(num_players)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        current_agent_index = 0
        num_adversaries = arglist.num_adversaries

        print('Starting iterations...')
        while True:

            agent = env.agents[current_agent_index]
            trainer = trainers[current_agent_index]
            player_key = env.agent_selection
            obs = env.observe(agent=agent).flatten()
            action_probability = trainer.action(obs)
            action = np.random.choice(a=np.linspace(0,len(action_probability)-1, num=len(action_probability), dtype=int), size=1, p=action_probability)[0]
            obs_n = env.observe(agent).flatten()
            env.step(action)
            new_obs_n, rew_n, done_n, info_n = env.observe(agent).flatten(), env.rewards, env.dones, env.infos
            player_info = info_n.get(player_key)

            rew_array = rew_n.values()

            episode_step += 1
            done = all(done_n)
            terminal = False #(episode_step >= arglist.max_episode_len)

            # experience(self, obs, act, mask, rew, new_obs, done)
            trainer.experience(obs_n, action_probability, None, [rew_n.get(player_key)], new_obs_n, done_n.get(player_key))

            for i, rew in enumerate(rew_array):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

                if(loss is not None and agent.sleep_regimen and agent.agent_mic != 0 and train_step % 100 == 0): # Change sleep frequency here if desired
                    original_policy_loss = loss[1]
                    new_loss = agent.update(trainers, train_step, sleeping=True)[1]
                    sleep_iteration = 0
                    while((sleep_iteration < 10) and (new_loss < original_policy_loss * 1.05)):
                        new_loss = agent.update(trainers, train_step, sleeping=True)[1]
                        sleep_iteration += 1 
                        #print("sleep walking")

            # save model, display training output
            if done and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                print(arglist.plots_dir)
                print(arglist.exp_name)

                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break
            
            current_agent_index += 1
            if(current_agent_index > num_players - 1):
                current_agent_index = 0


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

