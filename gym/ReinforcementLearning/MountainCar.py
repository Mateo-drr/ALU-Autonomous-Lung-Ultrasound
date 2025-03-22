# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:13:17 2024

@author: Mateo-drr
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math
import matplotlib.pyplot as plt
import matplotlib


env = gym.make("MountainCar-v0",render_mode="human")
env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# done=False

# while not done:
#     action = 2
#     new_state, reward, done, tlimit, _ = env.step(action)
#     env.render()

# env.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer2b = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.leaky_relu(self.layer2b(x),inplace=True)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def plot_durations(simTimes,show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(simTimes, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def plot_rewards(reward_list, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(reward_list, dtype=torch.float)+200
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    
BATCH_SIZE = 32
GAMMA = 0.99
expRIni = 0.9
expREnd = 0.05
expDecay = 1000
TAU = 0.005
LR = 1e-4
num_episodes=600

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

#In DQN we use two identical networks
#the policy_net is the one being trained to select the actions to do
policy_net = DQN(n_observations, n_actions).to(device)
#the target_net is a snapshot network, ie it acts as a  delayed network
#that the policy net uses to calculate the loss.
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps = 0
simTimes = []
simRewds = []

print('beginning...')

for i in range(0,num_episodes):
    
    print('Episode', i)
    state, info = env.reset() #in this case it gives state = position in x , speed
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    #run the scene ie loop frames
    t=0
    totReward=0
    while True:
        
        '''
        Calculate next action
        '''
        
        threshold = random.random()
        explorationRate = expREnd+(expRIni-expREnd)*math.exp(-1.*steps/expDecay)
        
        if threshold > explorationRate:
            #Calculate the next move with the model
            qvalues = policy_net(state)
            #Get the move with highest score (idk why max(1) gives indx)
            action = qvalues.max(1).indices
        else:
            #Pick a random next move ie explore
            action = torch.tensor([env.action_space.sample()])
        
        '''
        Make a move
        '''
        
        newState, reward, done, tlimit, info = env.step(action.item())
        done = done or tlimit
        
        #Fix reward system
        
        reward += [1 if abs(state[0][1].item()) >= 0.005 else 0][0] #5*(state[0][0].item() - newState[0].item())**2
        totReward += reward
        
        '''
        Store the moves
        '''
        
        if done:
            newState = None
        else:
            newState = torch.tensor(newState, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state.to(device),
                    action.to(device),
                    newState,
                    torch.tensor([reward], device=device))
        # Set newState as current state
        state = newState
        
        '''
        Format the data
        '''
        
        #check if there is enough data in the memory
        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            #This converts batch-array of Transitions
            # to Transition of batch-arrays.
            # go from [Transitions,Transitions] to Transitions[]
            batch = Transition(*zip(*batch)) 
            
            #Get a mask of final and non final states
            nonFinalMask = []
            for s in batch.next_state:
                if s is not None:
                    nonFinalMask.append(True)
                else:
                    nonFinalMask.append(False)
            nonFinalMask = torch.tensor(nonFinalMask,device=device,dtype=torch.bool)
            
            #Get only the nonFinal ones
            nonFinal = torch.cat([s for s in batch.next_state if s is not None])

            #Turn the lists of tensors into tensors
            stateB = torch.cat(batch.state)
            actionB = torch.cat(batch.action).unsqueeze(1)
            rewardB = torch.cat(batch.reward)
            
            '''
            Calculate Q(s,a) <- r + w * max(Q(s+1,a')) ie the correct actions
            Basically calculate the labels from the snapshot network
            '''
            
            #Predict the actions and take the value predicted only of the correct action
            predActions = policy_net(stateB).gather(1, actionB)
            
            #Place to store the Q(s+1,a) values (next state)
            nextStateValues = torch.zeros(BATCH_SIZE,device=device)
            
            #Run snapshot network
            with torch.no_grad():
                # Get the Q-values for all possible actions in the next states
                nextStateQvals = target_net(nonFinal)
            
                #Select and store max values
                nextStateValues[nonFinalMask] = nextStateQvals.max(1).values
            
            # Compute the expected Q values
            expQvals = (rewardB + (GAMMA * nextStateValues)).unsqueeze(1)
            
            '''
            Optimize the policy model
            '''
            
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(predActions, expQvals)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()
            
        '''
        Optimize the snapshot/target model θ′ ← τ θ + (1 −τ )θ′
        '''

        #Get the state dict of the nets
        targetSD = target_net.state_dict()
        policySD = policy_net.state_dict()
        
        #apply the formula to each weight
        for key in policySD:
            targetSD[key] = TAU*policySD[key] + (1-TAU)*targetSD[key]
            
        #load the modified weights into the network
        target_net.load_state_dict(targetSD)
            
        '''
        Check to stop scene loop
        '''
            
        if done:
            simTimes.append(t + 1)
            plot_durations(simTimes)
            
            simRewds.append(totReward)
            plot_rewards(simRewds)
            break    
            
        t+=1
            
            
env.close()            
            
            