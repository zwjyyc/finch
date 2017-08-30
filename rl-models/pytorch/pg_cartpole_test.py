from pg import PolicyGradient
import gym


def main():
    model = PolicyGradient(gym.make('CartPole-v0'))
    model.learn()
    model.play()

if __name__ == '__main__':
    main()
