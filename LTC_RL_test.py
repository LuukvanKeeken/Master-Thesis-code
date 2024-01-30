from datetime import date
import os
import random
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP, FullyConnected, Wiring
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque
import torch
from LTC_A2C import LTC_Network, CfC_Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_BP_agent_pole_length(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
        
    for i_episode in range(num_episodes):
        hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            policy_output, value, hidden_state = agent_net(state.float(), hidden_state)
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards




def train_agent(env, num_training_episodes, max_steps, agent_net, num_outputs, evaluation_seeds, i_run, selection_method = "100 episode average", gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_evaluation_episodes = 10, evaluate_every = 10):
    best_average = -np.inf
    best_average_after = np.inf
    scores = []
    smoothed_scores = []
    scores_window = deque(maxlen = 100)

    entropy_term = 0

    for episode in range(1, num_training_episodes + 1):
        hidden_state = None

        score = 0

        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(max_steps):
            # Feed the state into the network
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device)
            policy_output, value, hidden_state = agent_net(state.float(), hidden_state)

            # Get distribution over the action space
            policy_dist = torch.softmax(policy_output, dim = 1)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            # Sample an action from the distribution
            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            new_state, reward, done, _ = env.step(action)

            score += reward

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == max_steps - 1:
                new_state = torch.from_numpy(new_state)
                new_state = new_state.unsqueeze(0).to(device)
                _, Qval, hidden_state = agent_net(new_state.float(), hidden_state)
                Qval = Qval.detach().numpy()[0, 0]

                if ((selection_method == "evaluation") and (episode % evaluate_every == 0)):
                    evaluation_performance = np.mean(evaluate_BP_agent_pole_length(agent_net, env_name, num_evaluation_episodes, evaluation_seeds, 1.0))
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + '/checkpoint_CfC_A2C_{}.pt'.format(i_run))

                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, '. Model saved in folder best.')
                        return smoothed_scores, scores, best_average, best_average_after



                elif (selection_method == "100 episode average"):
                    scores_window.append(score)
                    scores.append(score)
                    smoothed_scores.append(np.mean(scores_window))

                    if smoothed_scores[-1] >= best_average:
                        best_average = smoothed_scores[-1]
                        best_average_after = episode
                        # SAVE MODEL HERE
                    
                    print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                    if episode % 100 == 0:
                        print("\rEpisode {}\tAverage Score: {:.2f}".
                            format(episode, np.mean(scores_window)))
                        
                break

        # Compute the Q-values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * Qval
            Qvals[t] = Qval

        # Update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        # 0.001 IS A MAGIC NUMBER!!
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()
    
    print(f'Best {selection_method}: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
    
    return smoothed_scores, scores, best_average, best_average_after




learning_rate = 0.0001
selection_method = "evaluation"
gamma = 0.99
training_method = "standard"


dirs = os.listdir('./LTC_A2C/training_results/')
if not any('a2c_result' in d for d in dirs):
    result_id = 1
else:
    results = [d for d in dirs if 'a2c_result' in d]
    result_id = len(results) + 1
d = date.today()
result_dir = 'LTC_A2C/training_results/LTC_a2c_result_' + str(result_id) + f'_{str(d.year)+str(d.month)+str(d.day)}_learningrate_{learning_rate}_selectiomethod_{selection_method}_gamma_{gamma}_trainingmethod_{training_method}'
os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))




env = gym.make('CartPole-v0')
evaluation_seeds = np.load('rstdp_cartpole_stuff/seeds/evaluation_seeds.npy')
training_seeds = np.load('rstdp_cartpole_stuff/seeds/training_seeds.npy')


best_average_after_all = []
for i in range(10):
    print(f"Run # {i}")
    seed = int(training_seeds[i])

    torch.manual_seed(seed)
    random.seed(seed)

    agent_net = CfC_Network(4, 64, 2, seed).to(device)
    optimizer = torch.optim.Adam(agent_net.parameters(), lr=learning_rate)

    smoothed_scores, scores, best_average, best_average_after = train_agent(env, 10000, 200, agent_net, 2, evaluation_seeds, i, selection_method = selection_method, gamma = gamma)
    best_average_after_all.append(best_average_after)


with open(f"{result_dir}/best_average_after.txt", 'w') as f:
    for i, best_episode in enumerate(best_average_after_all):
        f.write(f"{i}: {best_episode}\n")

    f.write(f"Average: {np.mean(best_average_after_all)}, std dev: {np.std(best_average_after_all)}")






