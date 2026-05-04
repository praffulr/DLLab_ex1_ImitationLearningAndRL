import sys

sys.path.append("./")

import os
import numpy as np
import gymnasium as gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state, _ = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminated, truncated, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminated or truncated)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminated or truncated or step > max_timesteps:
            print ("Episode Ended - #",step)
            print (terminated, truncated, step>max_timesteps)
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(
        tensorboard_dir,
        "CartPole",
        stats=["train/episode_reward", "train/a_0", "train/a_1"],
    )

    tensorboard_eval = Evaluation(
        tensorboard_dir,
        "CartPole",
        stats=["eval/episode_reward", "eval/a_0", "eval/a_1"],
    )

    # training
    best_eval_reward = 0
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "train/episode_reward": stats.episode_reward,
                "train/a_0": stats.get_action_usage(0),
                "train/a_1": stats.get_action_usage(1),
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % eval_cycle == 0:
           sum_eval_reward = 0
           for j in range(num_eval_episodes):
                ep_stats = run_episode(env, agent, deterministic=True, do_training=False)
                sum_eval_reward += ep_stats.episode_reward
                tensorboard_eval.write_episode_data(
                    i,
                    eval_dict={
                        "eval/episode_reward": ep_stats.episode_reward,
                        "eval/a_0": ep_stats.get_action_usage(0),
                        "eval/a_1": ep_stats.get_action_usage(1),
                    },
                )
        #store best eval model
        if i % eval_cycle == 0 and sum_eval_reward > best_eval_reward:
            print ("New Best eval reward - ", sum_eval_reward, best_eval_reward)
            best_eval_reward = sum_eval_reward
            agent.save(os.path.join(model_dir, "dqn_agent_cartpole_besteval.pt"))

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_cartpole.pt"))
        

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 10 episodes

    # You find information about cartpole in
    # https://gymnasium.farama.org/environments/classic_control/cart_pole/
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 490.0 over 100 consecutive trials.

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    state_dim = 4
    num_actions = 2

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    q = MLP(state_dim=4, action_dim=2)
    q_target = MLP(state_dim=4, action_dim=2)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    dqnAgent = DQNAgent(q, q_target, num_actions)
    # 3. train DQN agent with train_online(...)
    train_online(env, dqnAgent, num_episodes=200)
