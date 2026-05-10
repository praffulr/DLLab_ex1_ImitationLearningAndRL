import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import gymnasium as gym

import sys

sys.path.append(".")

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from test import run_episode, states_with_history






def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    # print("INSIDE READ DATA FUNCTION")
    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    # print("X valid shape: ", X_valid.shape)
    # print("y valid shape: ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    # Converting to Grayscale
    # print ("Before Preprocess-", np.max(X_train))

    X_train = rgb2gray(X_train) / 255.0
    X_valid = rgb2gray(X_valid) / 255.0


     # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    # append image history to first state

    if (history_length > 1):
        X_train = states_with_history(X_train, history_length=history_length)

        X_valid = states_with_history(X_valid, history_length=history_length)
    else:
        #Expand dims in X
        X_train = np.expand_dims(X_train, axis=1)
        X_valid = np.expand_dims(X_valid, axis=1)

    # #Convert to discrete actions - default - for single press
    # y_train = np.apply_along_axis(arr = y_train, func1d = action_to_id, axis = 1)
    # y_valid = np.apply_along_axis(arr = y_valid, func1d = action_to_id, axis = 1)

    #Convert to discrete actions - for multipress
    y_train = np.apply_along_axis(arr = y_train, func1d = action_to_id_custom, axis = 1)
    y_valid = np.apply_along_axis(arr = y_valid, func1d = action_to_id_custom, axis = 1)


    # print("INSIDE PREPROCESSING FUNCTION")
    # print("X train shape: ", X_train.shape)
    # print("y train shape: ", y_train.shape)
    # print("X valid shape: ", X_valid.shape)
    # print("y valid shape: ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    num_epochs,
    batch_size,
    lr,
    model_dir="./models/IL/",
    tensorboard_dir="./tensorboard",
    history_length = 1,
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")
    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent(lr=lr, history_length=history_length)

    tensorboard_eval = Evaluation(tensorboard_dir, "Imitation Learning", stats=["train/accuracy", "train/loss", "eval/loss", "eval/accuracy", "eval/mean_reward"])

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop
    

    X_train, y_train = convert(X_train), convert(y_train).to(torch.int64)
    X_valid, y_valid = convert(X_valid), convert(y_valid).to(torch.int64)

    # Create a Dataset class
    dataset = torch.utils.data.TensorDataset(X_train, y_train)

    #Training
    best_eval_reward = 0
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    for i in range(num_epochs):
        #Turn on training mo
        agent.net.train()
        print ("Epoch #", i)
        # Sample from Dataset using DataLoader
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for (j, (X_batch, y_batch)) in enumerate(dataLoader):
            # 1. Training per batch
            training_loss_batch = agent.update(X_batch, y_batch)
            # 2. Training/Validation accuracy
            if j % 20 == 0:
                print ("Train iteration for epoch #", i, " miniBatch#", j)
                # compute training/ validation accuracy and write it to tensorboard
                training_preds = agent.predict(X_batch)
                # training_acc = torch.sum(training_preds == y_batch)/len(y_train)
                training_loss = training_loss_batch
                validation_preds = agent.predict(X_valid)
                # print("Validation output size - ", validation_preds.shape)
                # print("Training Outputs shape - ", training_preds.shape)
                validation_loss_fxn = torch.nn.CrossEntropyLoss()
                validation_loss = validation_loss_fxn(validation_preds, y_valid)
                # validation_loss = torch.nn.MSELoss()(y_valid, validation_preds)

                
                # validation_acc = torch.sum(validation_preds==y_valid)/len(y_valid)

                # For Continuous data
                print ("Validation Loss - ", validation_loss.item())
                print("Training Loss - ", training_loss.item())


                # For Discrete data
                # print ("Validation Acc - ", validation_acc)
                # print("Training Acc - ", training_acc)
               

                tensorboard_eval.write_episode_data(
                    episode=i, eval_dict={
                        # "train/accuracy": training_acc,
                        # "eval/accuracy": validation_acc,
                        "train/loss": training_loss,
                        "eval/loss": validation_loss
                    })

        #Eval Cycle - Turn on Eval mode
        agent.net.eval()
        if (i+1) % eval_cycle == 0:
            mean_eval_reward = 0
            for j in range(num_eval_episodes):
                episode_reward = run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=history_length)
                mean_eval_reward += episode_reward
            #store best eval model
            mean_eval_reward /= num_eval_episodes
            print ("New Best eval reward - ", mean_eval_reward, best_eval_reward)
            tensorboard_eval.write_episode_data(
                    episode=i, eval_dict={
                        "eval/mean_reward": mean_eval_reward
                    })
            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                agent.save(os.path.join(model_dir, "bc_agent_carracing_besteval.pt"))


        # # TODO: save your agent 
        # # store model.
        # if i % eval_cycle == 0 or i >= (num_epochs - 1):
        #     agent.save(os.path.join(model_dir, "bc_agent_carracing.pt"))
        #store checkpoint model
        if i % eval_cycle == 0 or (i >= num_epochs - 1):
            agent.save(os.path.join(model_dir, f"bc_agent_carracing_{i}_.pt"))      
            agent.save(os.path.join(model_dir, "bc_agent_carracing.pt"))


if __name__ == "__main__":

    #Hyper-Parameters
    eval_cycle = 20
    num_eval_episodes = 10
    num_epochs = 100
    batch_size = 64
    lr = 1e-4
    history_length = 1


    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length
    )

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, num_epochs = num_epochs, batch_size=batch_size, lr=lr, history_length = history_length)
