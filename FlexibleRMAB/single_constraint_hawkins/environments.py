import numpy as np
import gym
import evaluation

class dropOutState(gym.Env):
    def __init__(self, N, B, start_state, P_noise=False):
        
        self.N = N
        self.B = B
        self.S = 3
        self.current_state = start_state
        self.T_one, self.T_two, self.R, self.C = self.get_experiment(P_noise)

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
                    noise1 = (np.random.random_sample(2)/4)
                    noise2 = (np.random.random_sample(2)/4)
                    noise = np.concatenate([noise1,noise2]) #p(1,0,0), p(2,0,1), p(1,1,1), p(2,1,1)
                    # p(2,0,1) > p(2,1,1)
                    # p(1,0,0) > p(1,1,1)
                    if (noise[1]>noise[3]) and (noise[0]>noise[2]):
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
            T_one = np.array(T_one)

        # two-step transition probabilities
        T_two = []
        for i in range(self.N):
            P_i = T_one[i]
            P_i = np.swapaxes(P_i, 0, 1)
            P_i0 = P_i[0]

            T0 = np.matmul(P_i0,P_i0) # action 0: (0,0)
            T1 = np.matmul(P1,P_i0) # action 1: (1,0)
            T2 = np.matmul(P_i0,P1) # action 2: (0,1)
            T3 = np.matmul(P1,P1) # action 3: (1,1)

            T_i = np.array([T0, T1, T2, T3])
            T_i = np.swapaxes(T_i, 0, 1)
            T_two.append(T_i)

        T_two = np.array(T_two)

        # three-step transition probabilities
        T0 = np.matmul(P0,P0,P0) # action 0: (0,0,0)
        T1 = np.matmul(P1,P0,P0) # action 1: (1,0,0)
        T2 = np.matmul(P0,P1,P0) # action 2: (0,1,0)
        T3 = np.matmul(P0,P0,P1) # action 3: (0,0,1)
        T4 = np.matmul(P1,P1,P0) # action 3: (1,1,0)
        T5 = np.matmul(P1,P0,P1) # action 3: (1,0,1)
        T6 = np.matmul(P0,P1,P1) # action 3: (0,1,1)
        T7 = np.matmul(P1,P1,P1) # action 3: (1,1,1)

        T_i = np.array([T0, T1, T2, T3, T4, T5, T6, T7])
        T_i = np.swapaxes(T_i, 0, 1)
        T_three = np.array([T_i for _ in range(self.N)])

        return T_one, T_two, R, C

    def onestep(self, actions):
        current_state = evaluation.nextState(actions,self.current_state, self.S, self.T_one)
        self.current_state = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

    def multiple_steps(self, size, actions, random_states):
        rewards = []
        states = []
        for i in range(size):
            np.random.set_state(random_states[i])
            new_state = evaluation.nextState(actions[i],self.current_state, self.S, self.T_one)
            reward = evaluation.getReward(new_state, self.R)
            states.append(new_state)
            rewards.append(reward)
            self.current_state = new_state  

        return states, rewards
    
    def twostep_window(self, actions):
        current_state = evaluation.nextState(actions,self.current_state_twostepwd, self.S, self.P)
        self.current_state_twostepwd = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

