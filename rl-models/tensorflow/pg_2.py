import tensorflow as tf
import numpy as np


class PolicyGradient:
    def __init__(self, env, n_in, hidden_net, n_out, lr=0.01, sess=tf.Session()):
        self.env = env
        self.n_in = n_in
        self.hidden_net = hidden_net
        self.n_out = n_out
        self.lr = lr
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_forward_path()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_in])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.int32, shape=[None])
    # end method build_forward_path


    def add_forward_path(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_in])
        hidden = self.hidden_net(self.X)
        self.logits = tf.layers.dense(hidden, self.n_out)
        outputs = tf.nn.softmax(self.logits)
        self.action = tf.multinomial(tf.log(outputs), num_samples=1)
    # end method build_forward_path


    def add_backward_path(self):
        # maximize (log_p * R) = minimize -(log_p * R)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
        self.loss = tf.reduce_mean(xentropy * self.rewards) # reward guided loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def learn(self, n_games_per_update=10, n_max_steps=1000, n_iterations=250, discount_rate=0.95):
        self.sess.run(tf.global_variables_initializer())
        ep_obs = []
        ep_actions = []

        for iteration in range(n_iterations):
            all_rewards = []
            for game in range(n_games_per_update):
                current_rewards = []
                obs = self.env.reset()
                for step in range(n_max_steps):
                    action_val = self.sess.run(self.action, {self.X: np.atleast_2d(obs)})
                    obs, reward, done, info = self.env.step(action_val[0][0])
                    ep_obs.append(obs)
                    ep_actions.append(action_val[0][0])
                    current_rewards.append(reward)
                    if done:
                        break
                all_rewards.append(current_rewards)

            all_rewards = self.discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
            ep_rewards = [reward for rewards in all_rewards for reward in rewards]
            _, loss = self.sess.run([self.train_op, self.loss], {self.X: np.vstack(ep_obs),
                                                                 self.rewards: np.array(ep_rewards),
                                                                 self.actions: np.array(ep_actions)})
            print("Iteration: %d, Loss: %.3f" % (iteration, loss))
            ep_obs = []
            ep_actions = []
    # end method learn


    def play(self):
        obs = self.env.reset()
        done = False
        count = 0
        while not done:
            self.env.render()
            action_val = self.sess.run(self.action, {self.X: np.atleast_2d(obs)})
            obs, reward, done, info = self.env.step(action_val[0][0])
            count += 1
        print(count)
    # end method play


    def discount_rewards(self, rewards, discount_rate):
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards
    # end method discount_rewards


    def discount_and_normalize_rewards(self, all_rewards, discount_rate):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]
    # end method discount_and_normalize_rewards
