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
                noise = (np.random.random_sample(2)/5)
                P_i0 = P0.copy()
                P_i0[1] = P_i0[1] +  np.array([noise[0], -1*noise[0], 0])
                P_i0[2] = P_i0[2] +  np.array([0, noise[1], -1*noise[1]])
                P_i = np.array([P_i0,P1])
                P_i = np.swapaxes(P_i, 0, 1)
                T_one.append(P_i)
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

        return T_one, T_two, R, C

    def onestep(self, actions):
        current_state = evaluation.nextState(actions,self.current_state, self.S, self.T_one)
        self.current_state = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

    def twostep(self, actions, random_states):

        actions_step1 = [(1 if x==1 or x==3 else 0) for x in actions]
        actions_step2 = [(1 if x==2 or x==3 else 0) for x in actions]

        np.random.set_state(random_states[0])
        state_step1 = evaluation.nextState(actions_step1,self.current_state_twostep, self.S, self.P)
        np.random.set_state(random_states[1])
        state_step2 = evaluation.nextState(actions_step2,state_step1, self.S, self.P)

        reward_step1 = evaluation.getReward(state_step1, self.R)
        reward_step2 = evaluation.getReward(state_step2, self.R)

        self.current_state_twostep = state_step2

        return (state_step1,state_step2),  (reward_step1,reward_step2)
    
    def twostep_window(self, actions):
        current_state = evaluation.nextState(actions,self.current_state_twostepwd, self.S, self.P)
        self.current_state_twostepwd = current_state
        reward = evaluation.getReward(current_state, self.R)

        return current_state, reward

