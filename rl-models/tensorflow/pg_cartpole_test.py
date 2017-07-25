from pg import PolicyGradient
import gym


def main():
    model = PolicyGradient()
    model.learn(gym.make("CartPole-v0"))
    model.play(gym.make("CartPole-v0"))

if __name__ == '__main__':
    main()
