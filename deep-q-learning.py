import gym
import math
import random
import numpy as np
import matplotlib

from matplotlib import pyplot as plt

from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



# ==================== Support Function/Classes ==================== 

# Q value calculator
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # return the predicted q value
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))

    # return the maximum q value predicted by target net among all possible next actions
    @staticmethod
    def get_next(target_net, next_states):

        # find location of all possible final states (where the epsidoe ends)
        # all black --> all 0 (if max value is 0 --> next is final state)
        final_state_locations = next_states.flatten(start_dim = 1) \
            .max(dim = 1)[0].eq(0).type(torch.bool)
        
        # find none final states
        non_final_state_locations = (final_state_locations == False)
        # get value of non final states
        non_final_states = next_states[non_final_state_locations]

        # find batch size by checking to see how many next states are in the tensor
        batch_size = next_states.shape[0]


        values = torch.zeros(batch_size).to(QValues.device)

        # set value at non final state locations to maximum predicted q value from the target net
        values[non_final_state_locations] = target_net(non_final_states).max(dim = 1)[0].detach()

        return values

# batch of experience --> experienc of batches
def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    # Experience(state = 1, action = 1, next state = 1, reward = 1)
    # Experience(state = 2, action = 2, next state = 2, reward = 2)
    # Experience(state = 3, action = 3, next state = 3, reward = 3)
    # |
    # |
    # V
    # Experience(state = (1, 2, 3), action = (1, 2, 3), next_state = (1, 2, 3), reward = (1, 2, 3))
    return (t1, t2, t3, t4)


# Deep Q Network 
class DQN(nn.Module): # extend the nn.Module class by pytorch
    # recieve screenshot images as input
    def __init__(self, img_height, img_width):
        super().__init__()

        # Two fully connected and one output
        # height * width * channel (3)
        self.fc1 = nn.Linear(in_features = img_height * img_width * 3, out_features = 24)
        # Dense layers
        self.fc2 = nn.Linear(in_features = 24, out_features = 32)
        # 2 output --> left or right 
        self.out = nn.Linear(in_features = 32, out_features = 2)
    
    # implements forward propagation 
    # tensor is passed in as t
    def forward(self, t):
        # flattened
        t = t.flatten(start_dim = 1)
        # pass through the first layer --> rectified linear unit (activation)
        t = F.relu(self.fc1(t))
        # pass through the second layer
        t = F.relu(self.fc2(t))
        # pass through the output layer
        t = self.out(t)
        return t

# Experience class
# tuple 'experience' to store experience replays
Experience = namedtuple (
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    # Set capacity
    def __init__(self, capacity):
        self.capacity = capacity
        # holds the stored experiences
        self.memory = []
        # count how many experiences are added
        self.push_count = 0

    # push memory into storage
    def push(self, experience):
        # if there is still space push memory
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # override and loop around
            self.memory[self.push_count % self.capcity] = experience
        self.push_count += 1

    # get a random memory
    def sample (self, batch_size):
        return random.sample(self.memory, batch_size)
    
    # return if can sample from memory or not
    # can sample if there is more memory exisitng than batch size
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# Epsilon greedy strategy (determine whether to explore or exploit)
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    # return the exploration rate based on the current step
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class Agent():
    # takes in the strategy (epsilon greedy) and number of actions that the agent can take
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        # cpu / gpu
        self.device = device
    
    # takes in the state and the policy network
    def select_action(self, state, policy_net):
        # get the exploration rate
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        # (EXPLORE) if the exploration rate is greater than a random number
        if rate > random.random():
            # random action
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device)
        # (EXPLOIT)
        else:
            # pick action with the highest q value from the policy network
            # turn off gradient tracking --> do not  train
            with torch.no_grad():
                return policy_net(state).argmax(dim = 1).to(device)


class CartPoleEnvManager():
    # pass in device 
    def __init__(self, device):
        self.device = device
        # get behind the scene dynamic
        self.env = gym.make('CartPole-v0').unwrapped
        # reset and get initial observations
        self.env.reset()
        self.current_screen = None
        # track if episode is ended
        self.done = False

    # wrapper functions
    def reset(self):
        self.env.reset()
        self.current_screen = None


    def close(self):
        self.env.close()
    
    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n
    
    # take action
    def take_action(self, action):
        # take action and update reward and see if episode has ended
        _, reward, self.done, _  = self.env.step(action.item()) 
        #.item() because action 
        # passed through will be a tensor (returns the value of the tensor)
        
        # returns reward wrapped in a tensor to be used later on
        return torch.tensor([reward], device = self.device)
        # keep data type consistent 
    
    # return if at the starting state of an episode and nothing is yet rendered
    def just_starting(self):
        return self.current_screen is None

    # return the current state of the environment in a processed image
    def get_state(self):
        # if at the start or end (there is no 'last screen')
        if self.just_starting() or self.done:
            # get the current processed screen
            self.current_screen = self.get_processed_screen()
            # get a black screen with the same shape
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        # in between episode
        else:
            # take the current scree and the different screen
            s1 = self.current_screen
            s2 = self.get_processed_screen()

            # set current_screen to the processed screen
            self.current_screen = s2
            # state is represented as the difference between the current screen and the previous screen
            return s2 - s1

    # return the height of the screen
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    # return the width of the screen
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    # process screen
    def get_processed_screen(self):
        # transpose into what pytorch dqn expects
        screen = self.render('rgb_array').transpose((2, 0, 1))

        # crop screen
        screen = self.crop_screen(screen)

        # return transformed screen
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # strip off top and bottom

        # only get 40 percent of screen height from top
        top = int(screen_height * 0.4)

        # only get 20 percent of screen height from bottom
        bottom = int(screen_height * 0.8)
        
        # from removed top 40% and bottom 20%
        screen = screen[:, top:bottom, :]

        return screen
    
    def transform_screen_data(self, screen):
        # convert to float --> rescale
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # convert to pytorch tensor
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])

        # add extra dimension (batch dimension)
        return resize(screen).unsqueeze(0).to(self.device)

# plot training
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)

    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])



# return the moving average
def get_moving_average(period, values):
    # convert to pytorch tensor
    values = torch.tensor(values, dtype = torch.float)
    # if atleast as large as period --> get moving average
    if len(values) >= period:
        # unfold tensor --> return tensor of all sizes equal to the period passed in
        moving_avg = values.unfold(dimension = 0, size = period, step = 1) \
            .mean(dim=1).flatten(start_dim=0) # take the mean and flatten all the tensor
        # concatenate tensor 
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        # return numpy array
        return moving_avg.numpy()
    else:
        # return 0
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()




# ==================== Main Program ==================== 
# Hyper parameter
batch_size = 256
# discount
gamma = 0.999
# epsilon greedy
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
# how frequently the target network is updated (episodes)
target_update = 10
# capacity of replay memory
memory_size = 100000
# learning rate
lr = 0.001
# total of episodes to run on
num_episodes = 1000

# use gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize
em = CartPoleEnvManager(device)
# use epsilon greedy strategy
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)


# create policy network
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

# create target network
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

# copy policy net to target net
target_net.load_state_dict(policy_net.state_dict())
# turn off training
target_net.eval()
# Adam optimizer
optimizer = optim.Adam(params = policy_net.parameters(), lr = lr)

# tore duration for plotting
episode_durations = []

# for each episode
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    #for each time step
    for timestep in count():
        # selects action based on strategy
        action = agent.select_action(state, policy_net)
        # take action and recieve reward and next state
        reward = em.take_action(action)
        next_state = em.get_state()
        # push experience to memory
        memory.push(Experience(state, action, next_state, reward))
        # transition to next state
        state = next_state

        if memory.can_provide_sample(batch_size):
            # get a sample of memory of size 'batch size'
            experiences = memory.sample(batch_size)

            # extract information
            states, actions, rewards, next_states = extract_tensors(experiences)

            # get q values
            current_q_values = QValues.get_current(policy_net, states, actions)

            # get q values of next states
            next_q_values = QValues.get_next(target_net, next_states)
            # Bellman optimality equation
            target_q_values = (next_q_values * gamma) + rewards


            # get loss (mean squared error)
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            # sets the gradients of all the weights and biases in the policy net to 0
            optimizer.zero_grad()

            # computes gradients
            loss.backward()
              
            # updates weights and biases
            optimizer.step()

        # if the episode ended
        if em.done:
            # plot and break out
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
    # if at 'target_update' set target network to the policy network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

# close
em.close()




