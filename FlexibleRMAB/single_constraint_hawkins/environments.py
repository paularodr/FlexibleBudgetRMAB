import numpy as np
import gym
import evaluation

class birthDeathProcess(gym.Env):
    def __init__(self, N, B, S, reward_ones=True):
        
        self.N = N
        self.B = B
        self.S = S
        self.current_state = np.array([S-1]*N)
        self.reward_ones = reward_ones
        self.T, self.R, self.C = self.get_experiment()
    
    def get_experiment(self):

        C = [0,1]
        if self.reward_ones:
            neg = [-1 for _ in range(int(self.S/2))]
            pos = [1 for _ in range(int(self.S/2))]
        else:
            neg = [-1*(self.S/2)+x for x in range(int(self.S/2))]
            pos = [x+1 for x in range(int(self.S/2))]

        if (self.S % 2) == 0:
            R = np.array([neg+pos for _ in range(self.N)])
        else:
            R = np.array([neg+[0]+pos for _ in range(self.N)])
        P = np.zeros((self.N,self.S,2,self.S))

        for i in range(self.N):
            for s in range(self.S):
                probs = np.random.random(2)

                if s != 0:
                    P[i,s,1,self.S - 1] = 1

                if s != 0:
                    P[i,s,0,s-1] = 0.45
                    P[i,s,0,s] = 1-0.45
                else:
                    P[i,s,0,s] = 1
                    P[i,s,1,s] = 1

        return P, R, C

    def onestep(self, actions):
        current_state = evaluation.nextState(actions,self.current_state, self.T)
        self.current_state = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

    def multiple_steps(self, size, actions, random_states):
        rewards = []
        states = []
        for i in range(size):
            np.random.set_state(random_states[i])
            new_state = evaluation.nextState(actions[i],self.current_state, self.T)
            reward = evaluation.getReward(new_state, self.R)
            states.append(new_state)
            rewards.append(reward)
            self.current_state = new_state  

        return states, rewards

class riskProneArms(gym.Env):
    def __init__(self, N, B, M):
        
        self.N = N
        self.B = B
        self.S = 2
        self.M = M
        self.current_state = np.array([1]*N)
        self.T, self.R, self.C = self.get_experiment()
    
    def get_experiment(self):

        C = [0,1]
        R = np.array([[0,1] for i in range(self.N)])


        #not act transition probability. RISK
        P0 = np.array([
            [0.9, 0.1],
            [0.3, 0.7]
            ])

        #act transition probability. RISK
        P1 = np.array([
            [0, 1],
            [0, 1]
            ])

        #not act transition probability. SELF CORRECTING
        Q0 = np.array([
            [0.1, 0.9],
            [0.3, 0.7]
            ])

        #act transition probability. SELF CORRECTING
        Q1 = np.array([
            [0, 1],
            [0, 1]
            ])

        P = []
        for i in range(self.N):
            if i <= np.round(self.N*self.M):
                Pi = [P0,P1]
            else:
                Pi = [Q0,Q1]
            Pi = np.swapaxes(Pi, 0, 1)
            P.append(Pi)

        P = np.array(P)
        return P, R, C

    def onestep(self, actions):
        current_state = evaluation.nextState(actions,self.current_state, self.T)
        self.current_state = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

    def multiple_steps(self, size, actions, random_states):
        rewards = []
        states = []
        for i in range(size):
            np.random.set_state(random_states[i])
            new_state = evaluation.nextState(actions[i],self.current_state, self.T)
            reward = evaluation.getReward(new_state, self.R)
            states.append(new_state)
            rewards.append(reward)
            self.current_state = new_state  

        return states, rewards





class dropOutState(gym.Env):
    def __init__(self, N, B, P_noise=False):
        
        self.N = N
        self.B = B
        self.S = 3
        self.current_state = np.array([2]*N)
        self.T, self.R, self.C = self.get_experiment(P_noise)

    def get_experiment(self,P_noise=False):

        C = [0,1,1,2]
        R = np.array([[0,1,1] for i in range(self.N)])
        # one-step transition probabilities
        #not act transition probability. SxS
        P0 = np.array([
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
            [0.0 ,0.2, 0.8]
            ])

        #act transition probability. SxS
        P1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
            ])

        P_i = np.array([P0,P1])
        P_i = np.swapaxes(P_i, 0, 1)
        if not P_noise:
            T_one = np.array([P_i for _ in range(self.N)])
        else:
            T_one = []
            for _ in range(self.N):
                stop = True
                while stop:
                    noise1 = [np.random.randint(75,90)/100,np.random.randint(15,30)/100]
                    #noise1 = np.array([0.8,0.2])
                    noise2 = np.random.randint(15,40,2)/100
                    noise2 = np.zeros(2)
                    noise = np.concatenate([noise1,noise2]) #p(1,0,0), p(2,0,1), p(1,1,1), p(2,1,1)
                    # p(2,0,1) > p(2,1,1)
                    if (noise[1]>noise[3]):
                        P_i0 = P0.copy()
                        P_i0[1] = np.array([noise[0], 1-noise[0], 0])
                        P_i0[2] = np.array([0, noise[1], 1-noise[1]])

                        P_i1 = P1.copy()
                        P_i1[1] = np.array([0,noise[2], 1-noise[2]])
                        P_i1[2] = np.array([0, noise[3], 1-noise[3]])

                        P_i = np.array([P_i0,P_i1])
                        P_i = np.swapaxes(P_i, 0, 1)
                        T_one.append(P_i)
                        stop = False
            T = np.array(T_one)


        return T, R, C

    def onestep(self, actions):
        current_state = evaluation.nextState(actions,self.current_state, self.T)
        self.current_state = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

    def multiple_steps(self, size, actions, random_states):
        rewards = []
        states = []
        for i in range(size):
            np.random.set_state(random_states[i])
            new_state = evaluation.nextState(actions[i],self.current_state, self.T)
            reward = evaluation.getReward(new_state, self.R)
            states.append(new_state)
            rewards.append(reward)
            self.current_state = new_state  

        return states, rewards
