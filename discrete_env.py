import gym

class Discrete_Env():
    """Base environment for discrete cases.

    Args:
        env (gym.Env): OpenAI Gym environment. Must have discrete action and observation spaces.
        gamma (float): discount factor.
    """
    def __init__(self, env, gamma=1):
        self.env = env
        #Check discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            print('Action space not discrete.')
            return
        self.A = env.action_space.n
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state_dimensions = [env.observation_space.n]
            self.single = True
        else:
            self.state_dimensions = []
            for i in range(len(env.observation_space)):
                if not isinstance(env.observation_space[i], gym.spaces.Discrete):
                    print('Observation space not discrete.')
                    return
                self.state_dimensions.append(env.observation_space[i].n)
            self.single = False
        self.S = 1
        for i in range(len(self.state_dimensions)):
            self.S *= self.state_dimensions[i]
        #Set other variables
        self.gamma = gamma
        print('State size:', self.S, 'with dimensions:', self.state_dimensions)

    def state_to_ind(self, state):
        if (self.single): return state
        ind = 0
        mul = 1
        for i in range(len(self.state_dimensions)):
            ind += state[-1-i] * mul
            mul *= self.state_dimensions[-1-i]
        return ind
