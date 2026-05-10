import numpy as np
import torch

#Changing the encoding to match with the action space in https://gymnasium.farama.org/environments/box2d/car_racing/#action-space
LEFT = 2
RIGHT = 1
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype("float32")


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 2
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 1
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0
    
def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 2
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 1
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def action_to_id_custom(a):
    """
    this method discretizes the actions.
    Important: Trying to make this work even with multiple presses
    """
    if a[0] == -1:
        return LEFT  # LEFT: 2
    elif a[0] == 1:
        return RIGHT  # RIGHT: 1
    elif a[1] == 1:
        return ACCELERATE  # ACCELERATE: 3
    elif a[2] == 0.2:
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8, max_brake =0.1):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.1])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.1])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, max_brake])
    else:
        return np.array([0.0, 0.0, 0.0])
    
def convert (nparray):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.tensor(nparray, dtype= torch.float32).to(device)

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
