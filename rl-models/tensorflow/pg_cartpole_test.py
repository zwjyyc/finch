from pg import PolicyGradients
import gym


def main():
    model = PolicyGradients()
    model.learn(gym.make("CartPole-v0"))
    model.play(gym.make("CartPole-v0"))

if __name__ == '__main__':
    main()
