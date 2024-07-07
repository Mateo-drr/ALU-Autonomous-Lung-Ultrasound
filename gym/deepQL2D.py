# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:48:46 2024

@author: Mateo-drr
"""

###############################################################################
#Gym
###############################################################################

from cstmGym import LungUS,Transition,ReplayMemory
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import wandb

torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark=True

#PARAMS
BATCH_SIZE = 256
GAMMA = 0.8 #1 = future rewards vs 0 = current rewards 
expRIni = 0.9 #initial exploration rate
expREnd = 0.05 #final exploration rate
expDecay = 1000 #exploration decay
TAU = 0.01 #snapshot network update rate
LR = 1e-3
num_episodes=600
wb=True

params = {
    'BATCH_SIZE': 128,
    'GAMMA': 0.8,  # 1 = future rewards vs 0 = current rewards 
    'expRIni': 0.9,  # initial exploration rate
    'expREnd': 0.05,  # final exploration rate
    'expDecay': 1000,  # exploration decay
    'TAU': 0.005,  # snapshot network update rate
    'LR': 1e-4,  # learning rate
    'num_episodes': 600
}

if wb:
    wandb.init(
        # set the wandb project where this run will be logged
        name='2DqL jump 256 0.8 nR T0.01',
        project="2DqLearning",
    
        # track hyperparameters and run metadata
        config=params
    )

current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = LungUS(path + '/numpy/', res=20)
env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN(nn.Module):

    def __init__(self, numpix, actions):
        super(DQN, self).__init__()
        self.c1 = nn.Conv2d(1, 2, 3,stride=1,padding=1,padding_mode='reflect')
        self.c2 = nn.Conv2d(8, 4, 3,stride=1,padding=1,padding_mode='reflect')
        self.c3 = nn.Conv2d(16, 8, 3,stride=1,padding=1,padding_mode='reflect')
        self.c4 = nn.Conv2d(32, 1, 3,stride=1,padding=1,padding_mode='reflect')
        self.px = nn.PixelUnshuffle(2)
        self.p2 = nn.PixelUnshuffle(16)
        self.l1 = nn.Linear(4096, 2048)
        self.l2 = nn.Linear(2048, actions)
        self.mish = nn.Mish(inplace=True)
        self.cx = nn.Conv2d(256, 1, 3,stride=1,padding=1,padding_mode='reflect')
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
        # self.autobot = nn.TransformerEncoder(encoder_layer, 12)

    def forward(self, xini):
        #[1,512,512]
        x = self.mish(self.px(self.c1(xini)))
        #[8,256,256]
        x = self.mish(self.px(self.c2(x)))
        #[16,128,128]
        x = self.mish(self.px(self.c3(x)))
        #[32,64,64]
        x = self.mish(self.px(self.c4(x)))
        #[4,32,32]
        
        x = x + 0.2*self.cx(self.p2(xini))
        
        x = torch.flatten(x,start_dim=1)
        #[4096]
        x = self.mish(self.l1(x))
        #[2048]
        # x = x.unsqueeze(1) #add sequence dim
        #[b,1,2048]
        #x = self.autobot(x).squeeze(1) #remove seq dim
        #[b,2048]
        x = self.l2(x)
        #[b,actions]
        return x.unsqueeze(0)
    
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
    rewards_t = torch.tensor(reward_list, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    #plt.yscale('log')  # Set the y-axis to a logarithmic scale
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

# Get number of actions from gym action space
actions = env.action_space.n
# Get the number of state observations
state = env.reset()
numpix = state.size

#In DQN we use two identical networks
#the policy_net is the one being trained to select the actions to do
policy_net = DQN(numpix, actions).to(device)
#the target_net is a snapshot network, ie it acts as a  delayed network
#that the policy net uses to calculate the loss.
target_net = DQN(numpix, actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps = 0
simTimes = []
simRewds = []

#DEBUG
# done=False
# action = 0
# while not done:
#     env.reset()
#     action+=1
#     print(action)
# #

for i in range(0,num_episodes):
    
    print('Episode', i)
    env.reset() 
    state, reward, done,_, info = env.step(0) #initial action, returns int8 np arr
    #format the data
    state = state.astype(float)/state.max()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
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
            qvalues = policy_net(state.to(device))
            #Get the move with highest score (idk why max(1) gives indx)
            action = qvalues[0].max(1).indices
        else:
            #Pick a random next move ie explore
            action = torch.tensor([env.action_space.sample()])
        
        '''
        Make a move
        '''
        newState, reward, done, tlimit, info = env.step(action.item())
        #env.render(mode='human')
        #format the data
        newState = newState.astype(float)/newState.max()
        
        done = done or tlimit
        totReward += reward
        
        '''
        Store the moves (timeout has to have > batch size)
        '''
        if done:
            newState = None
        else:
            newState = torch.tensor(newState, dtype=torch.float32, device='cpu').unsqueeze(0).detach()

        # Store the transition in memory (remove batch dim)
        memory.push(state[0].detach().cpu(),
                    action.detach().cpu(),
                    newState,
                    torch.tensor([reward], device='cpu'))
        # Set newState as current state (add batch dim)
        state = newState.unsqueeze(0) if newState is not None else None
        
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
            
            #Get only the nonFinal ones (add channel dim)
            nonFinal = torch.cat([s for s in batch.next_state if s is not None]).unsqueeze(1)

            #Turn the lists of tensors into tensors
            stateB = torch.cat(batch.state).unsqueeze(1).to(device) # add 1 channel
            actionB = torch.cat(batch.action).unsqueeze(1).to(device)
            rewardB = torch.cat(batch.reward).to(device)
            
            '''
            Calculate Q(s,a) <- r + w * max(Q(s+1,a')) ie the correct actions
            Basically calculate the labels from the snapshot network
            '''
            #Predict the actions and take the value predicted only of the correct action
            predActions = policy_net(stateB) #[1,32,numactions]
            predActions = predActions[0].gather(1, actionB)
            #Place to store the Q(s+1,a) values (next state)
            nextStateValues = torch.zeros(BATCH_SIZE,device=device)
            
            #Run snapshot network
            with torch.no_grad():
                # Get the Q-values for all possible actions in the next states
                nextStateQvals = target_net(nonFinal.to(device))
            
                #Select and store max values
                nextStateValues[nonFinalMask] = nextStateQvals[0].max(1).values
            
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
            #plot_durations(simTimes)
            
            simRewds.append(totReward)
            #plot_rewards(simRewds)
            
            if wb:
                try:
                    wandb.log({"duration": t+1,'rewards':totReward, 'loss':loss.item()})
                except:
                    wandb.log({"duration": t+1,'rewards':totReward})
            break    
            
        t+=1
        torch.cuda.empty_cache()


# done=False

# action = 0
# while not done:
#     new_state, reward, done,_, info = env.step(action)
#     env.render(mode='human')
#     print(str(np.round(reward,4)), str(env.action_map), info)
#     while True:
#         try:
#             action=int(input(': '))
#             break
#         except:
#             print('Select an action!')
        
#     print('moving...')

env.close()