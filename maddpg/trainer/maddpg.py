import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from scipy.stats import multivariate_normal

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, mut_inf_coef=0, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, mut_inf_coef=0, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

def m_train(act_space_n, m_index, m_func, optimizer, mut_inf_coef=0, grad_norm_clipping=None, scope="trainer", reuse=None, num_units=64):
    return 
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = 1
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        m_input = [1]
        m = m_func(m_input, 1, scope="m_func", num_units=num_units)[:,0]
        m_func_vars = U.scope_vars(U.absolute_scope_name("m_func"))

        m_loss = tf.reduce_mean(tf.square(m - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        m_reg = tf.reduce_mean(tf.square(m))
        loss = m_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        m_values = U.function(obs_ph_n + act_ph_n, m)

        # target network
        target_m = m_func(m_input, 1, scope="target_m_func", num_units=num_units)[:,0]
        target_m_func_vars = U.scope_vars(U.absolute_scope_name("target_m_func"))
        update_target_m = make_update_exp(m_func_vars, target_m_func_vars)

        target_m_values = U.function(obs_ph_n + act_ph_n, target_m)

        return train, update_target_m, {'m_values': m_values, 'target_m_values': target_m_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, agent_type="good", local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
        
        if(agent_type == "good"):
            self.mic = float(args.good_mic)
        else:
            self.mic = float(args.adv_mic)
        
        print("MIC for ", agent_type, " agent is ", self.mic)
        self.agent_type = agent_type

        # make a multivariate for each agent. 

        self.multivariate_mean = None
        self.multivariate_cov = None 
        self.margian_aprox_lr = 1e-2
        self.action_history = []

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            mut_inf_coef=self.mic ,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            mut_inf_coef=self.mic ,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        """
        self.m_train, self.m_update, self.m_debug = m_train(
            scope=self.name,
            act_space_n=act_space_n,
            m_index=agent_index,
            m_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            mut_inf_coef=self.mic ,
            grad_norm_clipping=0.5,
            num_units=args.num_units
        )
        """
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
    
    def sleep_regimen(self):
        return self.args.sleep_regimen
    
    def agent_mic(self):
        return self.mic

    def action(self, obs):
        action = self.act(obs[None])[0]

        if(len(self.replay_buffer) > self.max_replay_buffer_len): # dont add random actions to action history
            self.action_history.append(action)

        if(self.mic > 0 and len(self.action_history) >= 100):
            actions = np.stack(self.action_history)
            act_mu = actions.mean(axis=0)
            act_std  = actions.std(axis=0)

            if(self.multivariate_mean is None):
                self.multivariate_mean = act_mu
            else:
                previous_mean = self.multivariate_mean
                self.multivariate_mean = ((1 - self.margian_aprox_lr) * self.multivariate_mean) + (self.margian_aprox_lr * act_mu)
            
            if(self.multivariate_cov is None):
                self.multivariate_cov = np.diag(act_std)
            else:
                cov = (self.margian_aprox_lr * np.diag(act_std) + (1 - self.margian_aprox_lr) * self.multivariate_cov)
                mom_1 = (self.margian_aprox_lr * np.square(np.diag(act_mu))) + ((1 - self.margian_aprox_lr) * np.square(np.diag(previous_mean)))
                mom_2 = np.square((self.margian_aprox_lr * np.diag(act_mu)) + (1 - self.margian_aprox_lr)*np.diag(previous_mean))
                self.multivariate_cov = cov + mom_1 - mom_2
        
        if(len(self.action_history) > 100):
            self.action_history.pop(0)

        return action

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, sleeping=False):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        
        mir_penalty = 0
        if(self.mic > 0 and (not self.args.sleep_regimen or (self.args.sleep_regimen and sleeping))): # If sleep regimen is on, only use mic when sleeping
            try:
                multivar = multivariate_normal(self.multivariate_mean, self.multivariate_cov)
                logp_phi = multivar.logpdf(act) 
                logp_phi = logp_phi.reshape(self.args.batch_size, )

                p_phi = multivar.pdf(act) 
                p_phi = p_phi.reshape(self.args.batch_size, )

                action_mean = np.mean(act, axis=0)
                action_std = np.std(act, axis=0)
                action_cov = np.diag(action_std)
                policy_multivar = multivariate_normal(action_mean, action_cov)
                logp_pi = policy_multivar.logpdf(act) 
                logp_pi = logp_pi.reshape(self.args.batch_size, )

                p_pi = policy_multivar.pdf(act) 
                p_pi = p_pi.reshape(self.args.batch_size, )

                phi_entropy = -1 * np.sum(logp_phi * p_phi)
                pi_entropy = -1 * np.sum(logp_pi * p_pi)

                mir_penalty = self.mic * (phi_entropy - pi_entropy)
            except:
                mir_penalty = 0
        
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += (rew - mir_penalty) + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
