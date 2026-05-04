import os
from datetime import datetime
import gymnasium as gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="human")

    # TODO: load DQN agent
    q = MLP(state_dim=4, action_dim=2)
    q_target = MLP(state_dim=4, action_dim=2)
    agent = DQNAgent(q, q_target, 2)
    model_dir = "./models"

    for model in {"dqn_agent_cartpole.pt", "dqn_agent_cartpole_besteval.pt"}:
        agent.load(os.path.join(model_dir, model))
        n_test_episodes = 15

        episode_rewards = []
        for i in range(n_test_episodes):
            print ("Episode #", i)
            stats = run_episode(
                env, agent, deterministic=True, do_training=False, rendering=True
            )
            episode_rewards.append(stats.episode_reward)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["episode_rewards"] = episode_rewards
        results["mean"] = np.array(episode_rewards).mean()
        results["std"] = np.array(episode_rewards).std()

        if not os.path.exists("./results"):
            os.mkdir("./results")

        fname = f"./results/cartpole_results_dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(results, f)

    env.close()
    print("... finished")
