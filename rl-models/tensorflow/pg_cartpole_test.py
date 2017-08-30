from pg import PolicyGradient
import tensorflow as tf
import gym


def main():
    model = PolicyGradient(gym.make('CartPole-v0'),
                           hidden_net = lambda x : tf.layers.dense(x, 4, tf.nn.relu))
    model.learn()
    model.play()

if __name__ == '__main__':
    main()
