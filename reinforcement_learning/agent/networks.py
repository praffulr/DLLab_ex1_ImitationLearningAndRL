import torch.nn as nn
import torch
import torch.nn.functional as F

"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class CNN(nn.Module):
    def __init__(self, history_length=0, n_classes=5):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        #Feature Extraction with Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11,11)
                                )
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11,11))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11,11)
                                )
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))


        #Fully Connected Parts
        self.fc1 = nn.Linear(in_features=64*64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=3)
        self.non_linearity = F.relu

    def forward(self, x):
        # TODO: compute forward pass
        #Convolution Layers
        x = self.non_linearity(self.conv1(x))
        # print ("After Conv1 - ", x.shape)
        x = self.non_linearity(self.conv2(x))
        # print ("After Conv2 - ", x.shape)
        x = self.non_linearity(self.conv3(x))
        # print ("After Conv3 - ", x.shape)
        x = self.non_linearity(self.conv4(x))
        # print ("After Conv4 - ", x.shape)

        #Flattening (N,C,H,W) to (N,C*H*W)
        x = torch.flatten(x, start_dim=1)
        # print ("After Flattening - ", x.shape)

        #FC layers
        x = self.non_linearity(self.fc1(x))
        # print ("After FC1 - ", x.shape)
        x = self.fc2(x)
        # print ("After FC2 - ", x.shape)

        #Applying activations on output layer to put data back into range
        steering = x[:, 0:1] # Need to be in [-1, +1]
        gas_brake = x[:, 1:] #Need to be in [0, 1]
        steering_activated = F.tanh(steering)
        # print(steering_activated.shape)
        gas_brake_activated = F.sigmoid(gas_brake)
        # print(gas_brake_activated.shape)
        x = torch.cat((steering_activated, gas_brake_activated), dim=1)

        # print ("After Final activations - ", x.shape)



        return x
