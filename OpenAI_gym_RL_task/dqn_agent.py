# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

import numpy as np
import random
from collections import namedtuple, deque

from dqn_model import QNetwork,QNetworkBN,QNetworkLN,QNetworkWN,QNetworkRN,QNetworkRLN,QNetworkRNLN,QNetwork_UAM,QNetworkLN_UAM,QNetworkRN_UAM,QNetworkRLN_UAM,QNetworkRNLN_UAM

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 50         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_UAM():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, action_dim=False, qnet='DQN'):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_dim = action_dim

        # Q-Network
        self.qnet = qnet
        if qnet == 'DQN':
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-BN':
            self.qnetwork_local = QNetworkBN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkBN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-LN':
            self.qnetwork_local = QNetworkLN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkLN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-WN':
            self.qnetwork_local = QNetworkWN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkWN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-RN':
            self.qnetwork_local = QNetworkRN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-RLN':
            self.qnetwork_local = QNetworkRLN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRLN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-RNLN':
            self.qnetwork_local = QNetworkRNLN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRNLN(state_size, action_size, seed).to(device)     
        elif qnet == 'DQN-RN-UAM':
            self.qnetwork_local = QNetworkRN_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRN_UAM(state_size, action_size, seed).to(device)     
        elif qnet == 'DQN-LN-UAM':
            self.qnetwork_local = QNetworkLN_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkLN_UAM(state_size, action_size, seed).to(device)     
        elif qnet == 'DQN-RLN-UAM':
            self.qnetwork_local = QNetworkRLN_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRLN_UAM(state_size, action_size, seed).to(device)     
        elif qnet == 'DQN-RNLN-UAM':
            self.qnetwork_local = QNetworkRNLN_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRNLN_UAM(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-UAM':
            self.qnetwork_local = QNetwork_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork_UAM(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.x_history_stack_local=None
        self.COMP_stack_local=None
        self.x_history_stack_target=None
        self.COMP_stack_target=None

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        l_local,l_target = -1, -1

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                l_local,l_target = self.learn(experiences, GAMMA)
        return l_local,l_target

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            if self.qnet == 'DQN' or self.qnet == 'DQN-BN' or self.qnet == 'DQN-LN' or self.qnet == 'DQN-WN':
                action_values = self.qnetwork_local(state)
            elif self.qnet in ['DQN-UAM','DQN-RN-UAM','DQN-LN-UAM','DQN-RLN-UAM','DQN-RNLN-UAM']:
                action_values,self.x_history_stack_local,self.COMP_stack_local, l = self.qnetwork_local(state,self.x_history_stack_local,self.COMP_stack_local)
            else:
                action_values,self.x_history_stack_local,self.COMP_stack_local = self.qnetwork_local(state,self.x_history_stack_local,self.COMP_stack_local)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if self.action_dim:
            return action_values.cpu().data.numpy()[0]
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.qnet == 'DQN' or self.qnet == 'DQN-BN' or self.qnet == 'DQN-LN' or self.qnet == 'DQN-WN':
            if self.action_dim:
                Q_targets_next = self.qnetwork_target(next_states).detach()
            else:       
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        elif self.qnet in ['DQN-UAM','DQN-RN-UAM','DQN-LN-UAM','DQN-RLN-UAM','DQN-RNLN-UAM']:
            Q_targets_next,self.x_history_stack_target,self.COMP_stack_target, l_target = self.qnetwork_target(next_states,self.x_history_stack_target,self.COMP_stack_target)
            if self.action_dim:
                Q_targets_next = Q_targets_next.detach()
            else:
                Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)            	
        else:
            Q_targets_next,self.x_history_stack_target,self.COMP_stack_target = self.qnetwork_target(next_states,self.x_history_stack_target,self.COMP_stack_target)
            if self.action_dim:
                Q_targets_next = Q_targets_next.detach()
            else:
                Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
      
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        # Get max predicted Q values (for next states) from target model
        if self.qnet == 'DQN' or self.qnet == 'DQN-BN' or self.qnet == 'DQN-LN' or self.qnet == 'DQN-WN':
            if self.action_dim:
                Q_expected = self.qnetwork_local(states)
            else:
                Q_expected = self.qnetwork_local(states).gather(1, actions)
        elif self.qnet in ['DQN-UAM','DQN-RN-UAM','DQN-LN-UAM','DQN-RLN-UAM','DQN-RNLN-UAM']:
            Q_expected,self.x_history_stack_local,self.COMP_stack_local,l_local = self.qnetwork_local(states,self.x_history_stack_local,self.COMP_stack_local)
            if not self.action_dim:
                Q_expected = Q_expected.gather(1, actions)
        else:
            Q_expected,self.x_history_stack_local,self.COMP_stack_local = self.qnetwork_local(states,self.x_history_stack_local,self.COMP_stack_local)
            if not self.action_dim:
                Q_expected = Q_expected.gather(1, actions)
           
        # Get expected Q values from local model
        

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return l_local, l_target                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, action_dim=False, qnet='DQN'):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.action_dim = action_dim

        # Q-Network
        self.qnet = qnet
        if qnet == 'DQN':
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-BN':
            self.qnetwork_local = QNetworkBN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkBN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-LN':
            self.qnetwork_local = QNetworkLN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkLN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-WN':
            self.qnetwork_local = QNetworkWN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkWN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-RN':
            self.qnetwork_local = QNetworkRN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-RLN':
            self.qnetwork_local = QNetworkRLN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRLN(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-RNLN':
            self.qnetwork_local = QNetworkRNLN(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRNLN(state_size, action_size, seed).to(device)     
        elif qnet == 'DQN-RN-UAM':
            self.qnetwork_local = QNetworkRN_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetworkRN_UAM(state_size, action_size, seed).to(device)
        elif qnet == 'DQN-UAM':
            self.qnetwork_local = QNetwork_UAM(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork_UAM(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.x_history_stack_local=None
        self.COMP_stack_local=None
        self.x_history_stack_target=None
        self.COMP_stack_target=None

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            if self.qnet == 'DQN' or self.qnet == 'DQN-BN' or self.qnet == 'DQN-LN' or self.qnet == 'DQN-WN':
                action_values = self.qnetwork_local(state)
            elif self.qnet == 'DQN-UAM' or self.qnet == 'DQN-RN-UAM':
                action_values,self.x_history_stack_local,self.COMP_stack_local, l = self.qnetwork_local(state,self.x_history_stack_local,self.COMP_stack_local)
            else:
                action_values,self.x_history_stack_local,self.COMP_stack_local = self.qnetwork_local(state,self.x_history_stack_local,self.COMP_stack_local)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if self.action_dim:
            return action_values.cpu().data.numpy()[0]
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.qnet == 'DQN' or self.qnet == 'DQN-BN' or self.qnet == 'DQN-LN' or self.qnet == 'DQN-WN':
            if self.action_dim:
                Q_targets_next = self.qnetwork_target(next_states).detach()
            else:
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        elif self.qnet == 'DQN-UAM' or self.qnet == 'DQN-RN-UAM':
            Q_targets_next,self.x_history_stack_target,self.COMP_stack_target, l = self.qnetwork_target(next_states,self.x_history_stack_target,self.COMP_stack_target)
            if self.action_dim:
                Q_targets_next = Q_targets_next.detach()
            else:
                Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)
        else:
            Q_targets_next,self.x_history_stack_target,self.COMP_stack_target = self.qnetwork_target(next_states,self.x_history_stack_target,self.COMP_stack_target)
            if self.action_dim:
                Q_targets_next = Q_targets_next.detach()
            else:
                Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
      
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        # Get max predicted Q values (for next states) from target model
        if self.qnet == 'DQN' or self.qnet == 'DQN-BN' or self.qnet == 'DQN-LN' or self.qnet == 'DQN-WN':
            if self.action_dim:
                Q_expected = self.qnetwork_local(states)
            else:
                Q_expected = self.qnetwork_local(states).gather(1, actions)
        elif self.qnet == 'DQN-UAM' or self.qnet == 'DQN-BN-UAM':
            Q_expected,self.x_history_stack_local,self.COMP_stack_local,l = self.qnetwork_local(states,self.x_history_stack_local,self.COMP_stack_local)
            if not self.action_dim:
                Q_expected = Q_expected.gather(1, actions)
        else:
            Q_expected,self.x_history_stack_local,self.COMP_stack_local = self.qnetwork_local(states,self.x_history_stack_local,self.COMP_stack_local)
            if not self.action_dim:
                Q_expected = Q_expected.gather(1, actions)
           
        # Get expected Q values from local model
        

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)