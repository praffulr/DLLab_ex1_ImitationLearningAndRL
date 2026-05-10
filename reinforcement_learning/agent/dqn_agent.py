import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def convert (nparray):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.tensor(nparray, dtype= torch.float32).to(device)

class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        history_length=0,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        # self.Q = Q.cuda()
        # self.Q_target = Q_target.cuda()

        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        # self.loss_function = torch.nn.MSELoss()
        self.loss_function = torch.nn.HuberLoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update:
        #       2.1 compute td targets and loss
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)

        # 1 - Replay Buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        #2.1 - TD_target
        #2.2 - update Q

        #2.0 Sample batches from Replay Buffer
        batch = self.replay_buffer.next_batch(batch_size=64)
        batch = tuple(map(convert, batch))
        batch_states,batch_actions,batch_next_states,batch_rewards,batch_dones = batch


        #Estimates
        action_ids = batch_actions.unsqueeze(dim=1).to(torch.int64)
        # print ("In DQN Train -",batch_states.shape)
        q_values = self.Q(batch_states.float())
        # print ("Q_values shape - ", q_values.shape)
        # print ("Batch_actions shape - ", batch_actions.shape)
        # print ("Action_ids shape - ", action_ids.shape)
        # estimates = q_values[:, batch_actions]
        # print ("Estimates shape - ", estimates.shape)
        # print ("Action IDs - ", action_ids)
        # print ("Q values - ", q_values)

        estimates = torch.gather(input = q_values, dim = 1, index = action_ids).squeeze()



        #Targets
        best_q_values = torch.max(self.Q_target(batch_next_states), axis = 1)[0]
        # print ("best_q_values - ", best_q_values.shape)
        targets = batch_rewards + self.gamma*(1-batch_dones)*best_q_values

        loss = self.loss_function(estimates, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #2.3
        soft_update(self.Q_target, self.Q, self.tau)


    def act(self, state, deterministic, episode, num_episodes, action_usage=None):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """

        decaying_epsilon = self.epsilon*(1-episode/num_episodes)
        r = np.random.uniform()
        if deterministic or r > decaying_epsilon :
            pass
            # TODO: take greedy action (argmax)
            with torch.no_grad():
                action_vals = self.Q(torch.from_numpy(np.expand_dims(state, 0)).to(self.device))
            action_id = torch.argmax(action_vals).item()
            # print ("Greedy action vals - ", action_vals)
            # print ("Greedy action_id - ", action_id)

        else:
            # pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            #Uniform Sampling
            # action_id = np.random.randint(low=0, high= self.num_actions)
            #Weighted Sampling
            # print ("Action usage in ep #",episode," is ", action_usage)
            if action_usage is None:
                #Uniformly random sampling
                action_id = np.random.randint(low=0, high= self.num_actions)
            else:
                #Weighted Sampling
                action_id = np.random.choice(self.num_actions, p=action_usage)
            # print ("Random action_id - ", action_id)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
