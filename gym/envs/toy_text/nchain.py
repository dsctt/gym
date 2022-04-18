import gym
from gym import spaces
from gym.utils import seeding


class NChainEnv(gym.Env):
    """n-Chain environment

    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward

    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.

    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.

    The observed state is the current state in the chain (0 to n-1).

    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """

    def __init__(self, n=5, slip=0.2, small=2, large=10, max_iters=1000):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.iters = 0
        self.max_iters = max_iters

        # Setup transitions and rewards
        self.P = {s: {a: [] for a in range(2)} for s in range(n)}
        for state in self.P.keys():
            d = {}
            for action in self.P[state].keys():
                res = []

                f_reward = 0
                next_state = state + 1

                if state == self.n-1:
                    f_reward = large
                    next_state = state

                if action == 0: # forward is chosen

                    t1 = (1-slip, next_state, f_reward, None)
                    t2 = (slip, 0, small, None)
                    res.append(t1)
                    res.append(t2)

                else: # backward is chosen

                    t1 = (1-slip, 0, small, None)
                    t2 = (slip, next_state, f_reward, None)
                    res.append(t1)
                    res.append(t2)

                d[action] = res
            
            self.P[state] = d

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large

        done = False
        self.iters += 1
        if self.iters == self.max_iters:
            done = True
            self.iters = 0

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state
