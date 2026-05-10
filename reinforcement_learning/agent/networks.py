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
    
    
class CNN0(nn.Module):
    def __init__(self, history_length=1, n_classes=5):
        super(CNN0, self).__init__()
        # TODO : define layers of a convolutional neural network
        #Feature Extraction with Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11,11))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11,11))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11,11))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))


        #Fully Connected Parts
        self.fc1 = nn.Linear(in_features=64*64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=n_classes)
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
    
class CNN1(nn.Module):
    def __init__(self, history_length=1, n_classes=5):
        super(CNN1, self).__init__()
        # TODO : define layers of a convolutional neural network

        #Feature Extraction with Convolution Layers
        #Conv
        self.conv1 = nn.Conv2d(in_channels=history_length, out_channels=20, kernel_size=5, padding=1, stride=3, bias = False)
        #BatchNorm
        self.bn1 = nn.BatchNorm2d(num_features=20)
        #Max Pool
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride = 2)
        #ReLu Activation
        self.relu1 = nn.ReLU(inplace=True)

        self.cnn_block1 = nn.Sequential(self.conv1, self.bn1, self.maxpool1, self.relu1)


        #Flatten
        self.flatten1 = nn.Flatten(start_dim=1) #flattens only the non-batch axes

        #Fully Connected Parts
        #For History length=4
        self.fc1 = nn.Linear(in_features=5120, out_features=256)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=256, out_features=16)
        self.act2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(in_features=16, out_features=3)
        
        self.fc_block = nn.Sequential(self.fc1, self.act1, self.fc2, self.act2, self.fc3)


    def forward(self, x):
        # print ("In CNN forward pass - ", x.shape)
        x = nn.Sequential(self.cnn_block1, self.flatten1, self.fc_block)(x)

        #Post-processing
        x = torch.cat((F.tanh(x[:,0:1]), F.sigmoid(x[:,1:])), dim = 1)
        return x

class CNN2(nn.Module):
    def __init__(self, history_length=1, n_classes=5):
        super(CNN2, self).__init__()
        # TODO : define layers of a convolutional neural network

        #Feature Extraction with Convolution Layers
        #Conv Block 1 #Output shape - (B, 32, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=history_length, out_channels=32, kernel_size=5, padding=1, stride=3, bias = False)
        #BatchNorm
        self.bn1 = nn.BatchNorm2d(num_features=32)
        #ReLu Activation
        self.relu1 = nn.ReLU(inplace=True)

        self.cnn_block1 = nn.Sequential(self.conv1, self.bn1, self.relu1)

        #Conv Block 2 #Output Shape - (B, 64, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias = False)
        #BatchNorm
        self.bn2 = nn.BatchNorm2d(num_features=64)
        #ReLu Activation
        self.relu2 = nn.ReLU(inplace=True)

        self.cnn_block2 = nn.Sequential(self.conv2, self.bn2, self.relu2)

        #Conv Block 3 #Output Shape - (B, 128, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias = False)
        #BatchNorm
        self.bn3 = nn.BatchNorm2d(num_features=128)
        #ReLu Activation
        self.relu3 = nn.ReLU(inplace=True)

        self.cnn_block3 = nn.Sequential(self.conv3, self.bn3, self.relu3)

        #Flatten
        self.flatten1 = nn.Flatten(start_dim=1) #flattens only the non-batch axes

        #Fully Connected Parts
        self.fc1 = nn.Linear(in_features=128*64, out_features=256)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=256, out_features=16)
        self.act2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(in_features=16, out_features=n_classes)
        
        self.fc_block = nn.Sequential(self.fc1, self.act1, self.fc2, self.act2, self.fc3)


    def forward(self, x):
        
        x = nn.Sequential(self.cnn_block1, self.cnn_block2, self.cnn_block3, self.flatten1, self.fc_block)(x)
        return x
    
# class CNNSB3(nn.Module):
#     def __init__(self, history_length=1, n_classes=5):
#         super(CNNSB3, self).__init__()
#         # TODO : define layers of a convolutional neural network

#         #Feature Extraction with Convolution Layers
#         #Conv
#         self.conv1 = nn.Conv2d(in_channels=history_length, out_channels=20, kernel_size=5, padding=1, stride=3, bias = False)
#         #BatchNorm
#         self.bn1 = nn.BatchNorm2d(num_features=20)
#         #Max Pool
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride = 2)
#         #ReLu Activation
#         self.relu1 = nn.ReLU(inplace=True)

#         self.cnn_block1 = nn.Sequential(self.conv1, self.bn1, self.maxpool1, self.relu1)


#         #Flatten
#         self.flatten1 = nn.Flatten(start_dim=1) #flattens only the non-batch axes

#         #Fully Connected Parts
#         #For History length=4
#         self.fc1 = nn.Linear(in_features=5120, out_features=256)
#         self.act1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_features=256, out_features=16)
#         self.act2 = nn.ReLU(inplace=True)
#         self.fc3 = nn.Linear(in_features=16, out_features=3)
        
#         self.fc_block = nn.Sequential(self.fc1, self.act1, self.fc2, self.act2, self.fc3)


#     def forward(self, x):
#         # print ("In CNN forward pass - ", x.shape)
#         x = nn.Sequential(self.cnn_block1, self.flatten1, self.fc_block)(x)

#         #Post-processing
#         x = torch.cat((F.tanh(x[:,0:1]), F.sigmoid(x[:,1:])), dim = 1)
#         return x
