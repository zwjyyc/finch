from pg import PolicyGradient
import tensorflow as tf
import gym


def main():
    model = PolicyGradient(lambda x : tf.layers.dense(x, 4, tf.nn.relu))
    model.learn(gym.make("CartPole-v0"))
    model.play(gym.make("CartPole-v0"))

if __name__ == '__main__':
    main()
