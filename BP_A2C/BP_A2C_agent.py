from copy import deepcopy
import torch
import numpy as np
import gym
import random
import time
from collections import deque


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class A2C_Agent:

    def __init__(self, env_name, seed, agent_net, entropy_coef, value_pred_coef, gammaR, max_grad_norm, max_steps, batch_size,
                 num_training_episodes, optimizer, i_run, result_dir, best_model_selection_method,
                 num_evaluation_episodes, evaluation_seeds, max_reward, evaluate_every, network_type, continuous_actions = False):
        
        if batch_size > 1:
            print("Batch size larger than 1 not implemented yet. Program will continue with batch size set to 1.")
            

        self.env = gym.make(env_name)

        self.env.seed(seed)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.num_inputs = self.env.observation_space.shape[0]
        if hasattr(self.env.action_space, 'n'):     
            self.num_outputs = self.env.action_space.n
        else:
            self.num_outputs = self.env.action_space.shape[0]

        self.env_name = env_name
        self.batch_size = 1 
        self.agent_net = agent_net
        self.entropy_coef = entropy_coef # coefficient for the entropy reward (really Simpson index concentration measure)
        self.value_pred_coef = value_pred_coef # coefficient for value prediction loss
        self.gammaR = gammaR # discounting factor for rewards
        self.max_grad_norm = max_grad_norm # maximum gradient norm, used in gradient clipping
        self.max_steps = max_steps # maximum length of an episode
        self.num_training_episodes = num_training_episodes
        self.optimizer = optimizer
        self.i_run = i_run
        self.result_dir = result_dir
        self.selection_method = best_model_selection_method
        self.num_evaluation_episodes = num_evaluation_episodes
        self.evaluation_seeds = evaluation_seeds
        self.max_reward = max_reward
        self.training_seed = seed
        self.evaluate_every = evaluate_every
        self.network_type = network_type
        self.continuous_actions = continuous_actions


        # Initialize Hebbian traces
        self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size).to(device)

        # Initialize hidden activations
        self.hidden_activations = self.agent_net.initialZeroState(self.batch_size).to(device)


    def train_agent(self, randomization_params = None, randomize_every = 5):

        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen = 100)

        entropy_term = 0


        for episode in range(1, self.num_training_episodes + 1):
            
            if randomization_params and episode % randomize_every == 0:
                env = gym.make(self.env_name)
                env = randomize_env_params(env, randomization_params)
            
            self.hidden_activations = self.agent_net.initialZeroState(self.batch_size)
            self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size)
            
            score = 0
            
            log_probs = []
            values = []
            rewards = []

            state = self.env.reset()
            for steps in range(self.max_steps):
                # Feed the state into the network
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(state.float(), [self.hidden_activations, self.hebbian_traces])
                
                # Get distribution over the action space
                policy_dist = torch.softmax(policy_output, dim = 1)
                value = value.detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 

                # Sample from distribution to select action
                action = np.random.choice(self.num_outputs, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                
                new_state, reward, done, _ = self.env.step(action)

                score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                
                if done or steps == self.max_steps-1:
                    new_state = torch.from_numpy(new_state)
                    new_state = new_state.unsqueeze(0)#.to(device) #This as well?
                    _, Qval, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(new_state.float(), [self.hidden_activations, self.hebbian_traces])
                    Qval = Qval.detach().numpy()[0,0]

                    if ((self.selection_method == "evaluation") and (episode % self.evaluate_every == 0)):
                        evaluation_performance = np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, self.num_evaluation_episodes, self.evaluation_seeds, 1.0))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after
                            
                    elif ((self.selection_method == "range_evaluation") and (episode % self.evaluate_every == 0)):
                        # pole_length_mods = [0.1, 0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 17.0, 20.0]
                        pole_length_mods = [0.55, 10.5]
                        eps_per_setting = 5
                        evaluation_performance = 0
                        for i, mod in enumerate(pole_length_mods):
                            # Get performance over one episode with this pole length modifier, 
                            # skip over the first i evaluation seeds so not all episodes have
                            # the same seed.
                            evaluation_performance += np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], mod))

                        evaluation_performance /= len(pole_length_mods)
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                        
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after


                    elif ((self.selection_method == "range_evaluation_all_params") and (episode % self.evaluate_every == 0)):
                        pole_length_mods = [0.55, 10.5]
                        pole_mass_mods = [3.0]
                        force_mag_mods = [0.6, 3.5]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        for i in range(total_eval_eps):
                            np.random.seed(self.evaluation_seeds[i+eps_per_setting-1])
                            pole_length_mod = np.random.choice(pole_length_mods)
                            pole_mass_mod = np.random.choice(pole_mass_mods)
                            force_mag_mod = np.random.choice(force_mag_mods)
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after
    
                    elif ((self.selection_method == "true_range_eval_all_params") and (episode % self.evaluate_every == 0)):
                        validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        current_np_seed = np.random.get_state()
                        current_r_seed = random.getstate()
                        for i in range(total_eval_eps):
                            np.random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            pole_length_range = random.choice(validation_ranges[0])
                            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                            force_mag_range = random.choice(validation_ranges[2])
                            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after

                        np.random.set_state(current_np_seed)
                        random.setstate(current_r_seed)

                    elif (self.selection_method == "100 episode average"):
                        scores_window.append(score)
                        scores.append(score)
                        smoothed_scores.append(np.mean(scores_window))

                        if smoothed_scores[-1] > best_average:
                            best_average = smoothed_scores[-1]
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                    self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))

                        print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                        if episode % 100 == 0:
                            print("\rEpisode {}\tAverage Score: {:.2f}".
                                format(episode, np.mean(scores_window)))
                        
                    break

                    # all_rewards.append(np.sum(rewards))
                    # all_lengths.append(steps)
                    # average_lengths.append(np.mean(all_lengths[-10:]))
                    # if episode % 10 == 0:                    
                    #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    # break
            
            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gammaR * Qval
                Qvals[t] = Qval
    
            #update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            # print(actor_loss, critic_loss, entropy_term)
            ac_loss = actor_loss + 0.5 * critic_loss + 0.001 * entropy_term
            
            self.optimizer.zero_grad()
            ac_loss.backward()
            # for name, param in self.agent_net.named_parameters():
            #     if param.grad is None:
            #         print(f"None gradient for {name}")
            #     else:
            #         print(f"Gradient for {name}")
            self.optimizer.step()

        print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        
        return smoothed_scores, scores, best_average, best_average_after


    def train_agent_discrete(self, randomization_params = None, randomize_every = 5):
        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen = 100)

        training_total_rewards = []
        training_losses = []
        validation_total_rewards = []
        validation_losses = []

        for episode in range(1, self.num_training_episodes + 1):
            
            if randomization_params and episode % randomize_every == 0:
                env = gym.make(self.env_name)
                env = randomize_env_params(env, randomization_params)
            
            self.hidden_activations = self.agent_net.initialZeroState(self.batch_size)
            self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size)
            
            score = 0
            
            log_probs = []
            values = []
            rewards = []
            entropies = []

            state = self.env.reset()
            for steps in range(self.max_steps):
                # Feed the state into the network
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(state.float(), [self.hidden_activations, self.hebbian_traces])
                
                policy_dist_old = torch.softmax(policy_output, dim = 1)
                policy_dist = torch.distributions.Categorical(logits=policy_output)

                


                # means, std_devs = policy_output
                # policy_dist = torch.distributions.Normal(means, std_devs)
                # value = value.detach().numpy()[0,0]
                # policy_dist_detached = torch.distributions.Normal(means.detach(), std_devs.detach())

                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)
                entropy = policy_dist.entropy().mean()

                # Get distribution over the action space
                # policy_dist = torch.softmax(policy_output, dim = 1)
                # value = value.detach().numpy()[0,0]
                # dist = policy_dist.detach().numpy() 

                # # Sample from distribution to select action
                # action = np.random.choice(self.num_outputs, p=np.squeeze(dist))
                # log_prob = torch.log(policy_dist.squeeze(0)[action])
                # entropy = -np.sum(np.mean(dist) * np.log(dist))
                
                new_state, reward, done, _ = self.env.step(action[0].numpy())

                score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)
                state = new_state
                
                if done or steps == self.max_steps-1:
                    training_total_rewards.append(score)
                    new_state = torch.from_numpy(new_state)
                    new_state = new_state.unsqueeze(0)#.to(device) #This as well?
                    _, Qval, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(new_state.float(), [self.hidden_activations, self.hebbian_traces])
                    Qval = Qval.detach().numpy()[0,0]

                    if ((self.selection_method == "evaluation") and (episode % self.evaluate_every == 0)):
                        evaluation_performance = np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, self.num_evaluation_episodes, self.evaluation_seeds, 1.0))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after
                            
                    elif ((self.selection_method == "range_evaluation") and (episode % self.evaluate_every == 0)):
                        # pole_length_mods = [0.1, 0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 17.0, 20.0]
                        pole_length_mods = [0.55, 10.5]
                        eps_per_setting = 5
                        evaluation_performance = 0
                        for i, mod in enumerate(pole_length_mods):
                            # Get performance over one episode with this pole length modifier, 
                            # skip over the first i evaluation seeds so not all episodes have
                            # the same seed.
                            evaluation_performance += np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], mod))

                        evaluation_performance /= len(pole_length_mods)
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                        
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after


                    elif ((self.selection_method == "range_evaluation_all_params") and (episode % self.evaluate_every == 0)):
                        pole_length_mods = [0.55, 10.5]
                        pole_mass_mods = [3.0]
                        force_mag_mods = [0.6, 3.5]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        for i in range(total_eval_eps):
                            np.random.seed(self.evaluation_seeds[i+eps_per_setting-1])
                            pole_length_mod = np.random.choice(pole_length_mods)
                            pole_mass_mod = np.random.choice(pole_mass_mods)
                            force_mag_mod = np.random.choice(force_mag_mods)
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after
    
                    elif ((self.selection_method == "true_range_eval_all_params") and (episode % self.evaluate_every == 0)):
                        validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        current_np_seed = np.random.get_state()
                        current_r_seed = random.getstate()
                        for i in range(total_eval_eps):
                            np.random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            pole_length_range = random.choice(validation_ranges[0])
                            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                            force_mag_range = random.choice(validation_ranges[2])
                            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")
                        validation_total_rewards.append(evaluation_performance)
                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses

                        np.random.set_state(current_np_seed)
                        random.setstate(current_r_seed)
                    elif (self.selection_method == "exp_BW_validation" and (episode % self.evaluate_every == 0)):
                        evaluation_performance = np.mean(evaluate_BW(self.agent_net, self.env_name, 10, self.evaluation_seeds))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")
                        validation_total_rewards.append(evaluation_performance)
                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses

                    elif (self.selection_method == "100 episode average"):
                        scores_window.append(score)
                        scores.append(score)
                        smoothed_scores.append(np.mean(scores_window))

                        if smoothed_scores[-1] > best_average:
                            best_average = smoothed_scores[-1]
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                    self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))

                        print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                        if episode % 100 == 0:
                            print("\rEpisode {}\tAverage Score: {:.2f}".
                                format(episode, np.mean(scores_window)))
                        
                    break

                    # all_rewards.append(np.sum(rewards))
                    # all_lengths.append(steps)
                    # average_lengths.append(np.mean(all_lengths[-10:]))
                    # if episode % 10 == 0:                    
                    #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    # break
            
            # compute Q values
            values = torch.stack(values)
            Qvals = torch.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gammaR * Qval
                Qvals[t] = Qval
    
            #update actor critic
            # values = torch.FloatTensor(values)
            # values = torch.stack(values)
            Qvals = torch.FloatTensor(Qvals).requires_grad_(True)
            log_probs = torch.stack(log_probs).squeeze(1).sum(dim = -1)
            entropies = torch.stack(entropies).sum()
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            # print(actor_loss, critic_loss, entropy_term)
            ac_loss = actor_loss + self.value_pred_coef * critic_loss - self.entropy_coef * entropies
            training_losses.append(ac_loss)
            self.optimizer.zero_grad()
            ac_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), 50)
            # for name, param in self.agent_net.named_parameters():
            #     if param.grad is None:
            #         print(f"None gradient for {name}")
            #     else:
            #         print(f"Gradient for {name}")
            self.optimizer.step()

        print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        
        return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses




    def train_agent_cont(self, randomization_params = None, randomize_every = 5):

        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen = 100)

        training_total_rewards = []
        training_losses = []
        validation_total_rewards = []
        validation_losses = []

        for episode in range(1, self.num_training_episodes + 1):
            
            if randomization_params and episode % randomize_every == 0:
                env = gym.make(self.env_name)
                env = randomize_env_params(env, randomization_params)
            
            self.hidden_activations = self.agent_net.initialZeroState(self.batch_size)
            self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size)
            
            score = 0
            
            log_probs = []
            values = []
            rewards = []
            entropies = []

            state = self.env.reset()
            for steps in range(self.max_steps):
                # Feed the state into the network
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(state.float(), [self.hidden_activations, self.hebbian_traces])
                
                
                means, std_devs = policy_output
                policy_dist = torch.distributions.Normal(means, std_devs)
                # value = value.detach().numpy()[0,0]
                # policy_dist_detached = torch.distributions.Normal(means.detach(), std_devs.detach())

                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)
                entropy = policy_dist.entropy().mean()

                # Get distribution over the action space
                # policy_dist = torch.softmax(policy_output, dim = 1)
                # value = value.detach().numpy()[0,0]
                # dist = policy_dist.detach().numpy() 

                # # Sample from distribution to select action
                # action = np.random.choice(self.num_outputs, p=np.squeeze(dist))
                # log_prob = torch.log(policy_dist.squeeze(0)[action])
                # entropy = -np.sum(np.mean(dist) * np.log(dist))
                
                new_state, reward, done, _ = self.env.step(action[0].numpy())

                score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)
                state = new_state
                
                if done or steps == self.max_steps-1:
                    training_total_rewards.append(score)
                    new_state = torch.from_numpy(new_state)
                    new_state = new_state.unsqueeze(0)#.to(device) #This as well?
                    _, Qval, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(new_state.float(), [self.hidden_activations, self.hebbian_traces])
                    Qval = Qval.detach().numpy()[0,0]

                    if ((self.selection_method == "evaluation") and (episode % self.evaluate_every == 0)):
                        evaluation_performance = np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, self.num_evaluation_episodes, self.evaluation_seeds, 1.0))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after
                            
                    elif ((self.selection_method == "range_evaluation") and (episode % self.evaluate_every == 0)):
                        # pole_length_mods = [0.1, 0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 17.0, 20.0]
                        pole_length_mods = [0.55, 10.5]
                        eps_per_setting = 5
                        evaluation_performance = 0
                        for i, mod in enumerate(pole_length_mods):
                            # Get performance over one episode with this pole length modifier, 
                            # skip over the first i evaluation seeds so not all episodes have
                            # the same seed.
                            evaluation_performance += np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], mod))

                        evaluation_performance /= len(pole_length_mods)
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                        
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after


                    elif ((self.selection_method == "range_evaluation_all_params") and (episode % self.evaluate_every == 0)):
                        pole_length_mods = [0.55, 10.5]
                        pole_mass_mods = [3.0]
                        force_mag_mods = [0.6, 3.5]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        for i in range(total_eval_eps):
                            np.random.seed(self.evaluation_seeds[i+eps_per_setting-1])
                            pole_length_mod = np.random.choice(pole_length_mods)
                            pole_mass_mod = np.random.choice(pole_mass_mods)
                            force_mag_mod = np.random.choice(force_mag_mods)
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after
    
                    elif ((self.selection_method == "true_range_eval_all_params") and (episode % self.evaluate_every == 0)):
                        validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        current_np_seed = np.random.get_state()
                        current_r_seed = random.getstate()
                        for i in range(total_eval_eps):
                            np.random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            pole_length_range = random.choice(validation_ranges[0])
                            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                            force_mag_range = random.choice(validation_ranges[2])
                            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after

                        np.random.set_state(current_np_seed)
                        random.setstate(current_r_seed)
                    elif (self.selection_method == "exp_BW_validation" and (episode % self.evaluate_every == 0)):
                        evaluation_performance = np.mean(evaluate_BW(self.agent_net, self.env_name, 10, self.evaluation_seeds))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")
                        validation_total_rewards.append(evaluation_performance)
                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses

                    elif (self.selection_method == "100 episode average"):
                        scores_window.append(score)
                        scores.append(score)
                        smoothed_scores.append(np.mean(scores_window))

                        if smoothed_scores[-1] > best_average:
                            best_average = smoothed_scores[-1]
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                    self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))

                        print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                        if episode % 100 == 0:
                            print("\rEpisode {}\tAverage Score: {:.2f}".
                                format(episode, np.mean(scores_window)))
                        
                    break

                    # all_rewards.append(np.sum(rewards))
                    # all_lengths.append(steps)
                    # average_lengths.append(np.mean(all_lengths[-10:]))
                    # if episode % 10 == 0:                    
                    #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    # break
            
            # compute Q values
            values = torch.stack(values)
            Qvals = torch.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gammaR * Qval
                Qvals[t] = Qval
    
            #update actor critic
            # values = torch.FloatTensor(values)
            # values = torch.stack(values)
            Qvals = torch.FloatTensor(Qvals).requires_grad_(True)
            log_probs = torch.stack(log_probs).squeeze(1).sum(dim = -1)
            entropies = torch.stack(entropies).sum()
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            # print(actor_loss, critic_loss, entropy_term)
            ac_loss = actor_loss + self.value_pred_coef * critic_loss - self.entropy_coef * entropies
            
            self.optimizer.zero_grad()
            ac_loss.backward()
            # for name, param in self.agent_net.named_parameters():
            #     if param.grad is None:
            #         print(f"None gradient for {name}")
            #     else:
            #         print(f"Gradient for {name}")
            self.optimizer.step()

        print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        
        return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses




    def train_agent_new(self, randomization_params = None, randomize_every = 5):

        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen = 100)

        
        for episode in range(1, self.num_training_episodes + 1):
            if randomization_params and episode % randomize_every == 0:
                env = gym.make(self.env_name)
                env = randomize_env_params(env, randomization_params)
            
            self.hidden_activations = self.agent_net.initialZeroState(self.batch_size)
            self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size)
            
            score = 0
            
            log_probs = []
            entropy_vals = []
            values = []
            rewards = []

            state = self.env.reset()
            for steps in range(self.max_steps):
                # Feed the state into the network
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(state.float(), [self.hidden_activations, self.hebbian_traces])
                

                if self.continuous_actions:
                    # value_detached = value.detach().numpy()[0,0]
                    
                    means, std_devs = policy_output
                    dist = torch.distributions.Normal(means, std_devs)

                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy().mean()

                    action = action.detach().cpu().numpy()[0]

                else:
                    # Get distribution over the action space
                    policy_dist = torch.softmax(policy_output, dim = 1)
                    # value_detached = value.detach().numpy()[0,0]
                    dist = policy_dist.detach().cpu().numpy() 

                    # Sample from distribution to select action
                    action = np.random.choice(self.num_outputs, p=np.squeeze(dist))
                    log_prob = torch.log(policy_dist.squeeze(0)[action])
                    # entropy = -np.sum(np.mean(policy_dist) * np.log(policy_dist))
                    
                    entropy = -((policy_dist * torch.log(policy_dist)).sum(1).mean())
                
                new_state, reward, done, _ = self.env.step(action)

                score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_vals.append(entropy)
                state = new_state
                
                if done or steps == self.max_steps-1:
                    print(np.sum(rewards))
                    new_state = torch.from_numpy(new_state)
                    new_state = new_state.unsqueeze(0)#.to(device) #This as well?
                    _, Qval, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(new_state.float(), [self.hidden_activations, self.hebbian_traces])
                    Qval = Qval.detach().numpy()[0,0]

                    if ((self.selection_method == "evaluation") and (episode % self.evaluate_every == 0)):
                        evaluation_performance = np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, self.num_evaluation_episodes, self.evaluation_seeds, 1.0))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after
                            
                    elif ((self.selection_method == "range_evaluation") and (episode % self.evaluate_every == 0)):
                        # pole_length_mods = [0.1, 0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 17.0, 20.0]
                        pole_length_mods = [0.55, 10.5]
                        eps_per_setting = 5
                        evaluation_performance = 0
                        for i, mod in enumerate(pole_length_mods):
                            # Get performance over one episode with this pole length modifier, 
                            # skip over the first i evaluation seeds so not all episodes have
                            # the same seed.
                            evaluation_performance += np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], mod))

                        evaluation_performance /= len(pole_length_mods)
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                        
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after


                    elif ((self.selection_method == "range_evaluation_all_params") and (episode % self.evaluate_every == 0)):
                        pole_length_mods = [0.55, 10.5]
                        pole_mass_mods = [3.0]
                        force_mag_mods = [0.6, 3.5]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        for i in range(total_eval_eps):
                            np.random.seed(self.evaluation_seeds[i+eps_per_setting-1])
                            pole_length_mod = np.random.choice(pole_length_mods)
                            pole_mass_mod = np.random.choice(pole_mass_mods)
                            force_mag_mod = np.random.choice(force_mag_mods)
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after
    
                    elif ((self.selection_method == "true_range_eval_all_params") and (episode % self.evaluate_every == 0)):
                        validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        current_np_seed = np.random.get_state()
                        current_r_seed = random.getstate()
                        for i in range(total_eval_eps):
                            np.random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            pole_length_range = random.choice(validation_ranges[0])
                            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                            force_mag_range = random.choice(validation_ranges[2])
                            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after

                        np.random.set_state(current_np_seed)
                        random.setstate(current_r_seed)
                    elif (self.selection_method == "no_validation" and (episode % self.evaluate_every == 0)):
                        print("This is where validation would have been done.")



                    elif (self.selection_method == "100 episode average"):
                        scores_window.append(score)
                        scores.append(score)
                        smoothed_scores.append(np.mean(scores_window))

                        if smoothed_scores[-1] > best_average:
                            best_average = smoothed_scores[-1]
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                    self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))

                        print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                        if episode % 100 == 0:
                            print("\rEpisode {}\tAverage Score: {:.2f}".
                                format(episode, np.mean(scores_window)))
                        
                    break

                    # all_rewards.append(np.sum(rewards))
                    # all_lengths.append(steps)
                    # average_lengths.append(np.mean(all_lengths[-10:]))
                    # if episode % 10 == 0:                    
                    #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    # break
            
            # compute Q values
            
            values = torch.stack(values)
            Qvals = torch.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gammaR * Qval
                Qvals[t] = Qval
    
            # Qvals.requires_grad_(True)

            if self.continuous_actions:
                log_probs = torch.stack(log_probs).squeeze(1).sum(dim = -1)
            else:   
                log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            entropy_term = torch.stack(entropy_vals).sum()
            ac_loss = actor_loss + (self.value_pred_coef*critic_loss) - (self.entropy_coef * entropy_term)
            # print(actor_loss, self.value_pred_coef*critic_loss, self.entropy_coef * entropy_term)
            self.optimizer.zero_grad()
            ac_loss.backward()
            # for name, param in self.agent_net.named_parameters():
            #     if param.grad is None:
            #         print(f"None gradient for {name}")
            #     else:
            #         print(f"Gradient for {name}, mean: {param.grad.mean()}, std: {param.grad.std()}")
            self.optimizer.step()

        print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        
        return smoothed_scores, scores, best_average, best_average_after


    def train_agent_on_range(self, minimum, maximum):
        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen = 100)

        entropy_term = 0

        
        for episode in range(1, self.num_training_episodes + 1):
            self.hidden_activations = self.agent_net.initialZeroState(self.batch_size)
            self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size)
            
            score = 0
            
            log_probs = []
            values = []
            rewards = []

            modifier = np.random.uniform(minimum, maximum)
            self.env = gym.make(self.env_name)
            self.env.seed(self.training_seed + episode)
            self.env.unwrapped.length *= modifier

            state = self.env.reset()
            for steps in range(self.max_steps):
                # Feed the state into the network
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(state.float(), [self.hidden_activations, self.hebbian_traces])
                
                # Get distribution over the action space
                policy_dist = torch.softmax(policy_output, dim = 1)
                value = value.detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 

                # Sample from distribution to select action
                action = np.random.choice(self.num_outputs, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                
                new_state, reward, done, _ = self.env.step(action)

                score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                
                if done or steps == self.max_steps-1:
                    new_state = torch.from_numpy(new_state)
                    new_state = new_state.unsqueeze(0)#.to(device) #This as well?
                    _, Qval, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(new_state.float(), [self.hidden_activations, self.hebbian_traces])
                    Qval = Qval.detach().numpy()[0,0]

                    if ((self.selection_method == "evaluation") and (episode % 10 == 0)):
                        evaluation_performance = np.mean(evaluate_BP_agent_pole_length(self.agent_net, self.env_name, self.num_evaluation_episodes, self.evaluation_seeds, 1.0))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, '. Model saved in folder best.')
                            return smoothed_scores, scores, best_average, best_average_after
                            
                    elif ((self.selection_method == "true_range_eval_all_params") and (episode % self.evaluate_every == 0)):
                        validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                        eps_per_setting = 1
                        evaluation_performance = 0
                        total_eval_eps = 10
                        current_np_seed = np.random.get_state()
                        current_r_seed = random.getstate()
                        for i in range(total_eval_eps):
                            np.random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            random.seed((self.evaluation_seeds[i+eps_per_setting-1] + self.seed)%(2**32))
                            pole_length_range = random.choice(validation_ranges[0])
                            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                            force_mag_range = random.choice(validation_ranges[2])
                            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                            evaluation_performance += np.mean(evaluate_BP_agent_all_params(self.agent_net, self.env_name, eps_per_setting, self.evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                        evaluation_performance /= total_eval_eps
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance >= best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                        self.result_dir + f'/checkpoint_{self.network_type}_A2C_{self.i_run}.pt')
                            
                        if best_average == self.max_reward:
                            print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
                            best_average_after, f'. Model saved in folder {self.result_dir}')
                            return smoothed_scores, scores, best_average, best_average_after

                        np.random.set_state(current_np_seed)
                        random.setstate(current_r_seed)
    
                    elif (self.selection_method == "100 episode average"):
                        scores_window.append(score)
                        scores.append(score)
                        smoothed_scores.append(np.mean(scores_window))

                        if smoothed_scores[-1] > best_average:
                            best_average = smoothed_scores[-1]
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                    self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))

                        print("Episode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_window)), end='\r')

                        if episode % 100 == 0:
                            print("\rEpisode {}\tAverage Score: {:.2f}".
                                format(episode, np.mean(scores_window)))
                        
                    break

                    # all_rewards.append(np.sum(rewards))
                    # all_lengths.append(steps)
                    # average_lengths.append(np.mean(all_lengths[-10:]))
                    # if episode % 10 == 0:                    
                    #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    # break
            
            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gammaR * Qval
                Qvals[t] = Qval
    
            #update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = self.value_pred_coef * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + self.entropy_coef * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

        print(f'Best {self.selection_method}: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        
        return smoothed_scores, scores, best_average, best_average_after




    def fine_tune_agent(self, fine_tuning_episodes, eval_skip, fine_tuning_seeds, pole_length_modifier, n_evaluations, evaluation_seeds, max_reward):
        best_reward = -np.inf
        best_episode = 0
        best_weights = None
        entropy_term = 0

        # Check if pre-trained model already reaches perfect evaluation performance
        eval_rewards = evaluate_BP_agent_pole_length(self.agent_net, self.env_name, n_evaluations, evaluation_seeds, pole_length_modifier)
        avg_eval_reward = np.mean(eval_rewards)
        if avg_eval_reward >= max_reward:
            print("Maximum evaluation performance already reached before fine-tuning")
            best_weights = deepcopy(self.agent_net.state_dict())
            best_reward = avg_eval_reward
            print('\nBest individual stored after episode {:d} with reward {:6.2f}'.format(best_episode, best_reward))
            print()

            return best_weights, best_reward, best_episode


        # Evaluate after each episode of fine-tuning. Stop when perfect
        # evaluation performance is reached, or when there are no fine-
        # tuning episodes left.
        for i_episode in range(1, fine_tuning_episodes + 1):
            self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size).to(device)
            self.hidden_activations = self.agent_net.initialZeroState(self.batch_size).to(device)

            self.env.seed(int(fine_tuning_seeds[i_episode - 1]))

            score = 0

            log_probs = []
            values = []
            rewards = []


            state = self.env.reset()
            for steps in range(self.max_steps):
                # Feed the state into the network
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(state.float(), [self.hidden_activations, self.hebbian_traces])
                
                # Get distribution over the action space
                policy_dist = torch.softmax(policy_output, dim = 1)
                value = value.detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 

                # Sample from distribution to select action
                action = np.random.choice(self.num_outputs, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                
                new_state, reward, done, _ = self.env.step(action)

                score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                
                if done or steps == self.max_steps-1:
                    new_state = torch.from_numpy(new_state)
                    new_state = new_state.unsqueeze(0)#.to(device) #This as well?
                    _, Qval, (self.hidden_activations, self.hebbian_traces) = self.agent_net.forward(new_state.float(), [self.hidden_activations, self.hebbian_traces])
                    Qval = Qval.detach().numpy()[0,0]

            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.gammaR * Qval
                Qvals[t] = Qval
    
            #update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = self.value_pred_coef * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + self.entropy_coef * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()


            if (i_episode % eval_skip == 0):
                eval_rewards = evaluate_BP_agent_pole_length(self.agent_net, self.env_name, n_evaluations, evaluation_seeds, pole_length_modifier)
                avg_eval_reward = np.mean(eval_rewards)

                print("Episode: {:4d} -- Reward: {:7.2f} -- Best reward: {:7.2f} in episode {:4d}"\
                    .format(i_episode, avg_eval_reward, best_reward, best_episode), end='\r')    

                if avg_eval_reward > best_reward:
                    best_reward = avg_eval_reward
                    best_episode = i_episode
                    best_weights = deepcopy(self.agent_net.state_dict())
                    
                if best_reward >= max_reward:
                    break
  

        print('\nBest individual stored after episode {:d} with reward {:6.2f}'.format(best_episode, best_reward))
        print()
        return best_weights, best_reward, best_episode


def evaluate_BW(agent_net, env_name, num_episodes, evaluation_seeds):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)
            
        for i_episode in range(num_episodes):
            hebbian_traces = agent_net.initialZeroHebb(1)
            hidden_activations = agent_net.initialZeroState(1)
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0)#.to(device) #This as well?
                policy_output, value, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
                
                means, std_devs = policy_output
                
                # Get greedy action
                action = means
                

                state, r, done, _ = env.step(action[0].numpy())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards





def evaluate_BP_agent_pole_length(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
        
    for i_episode in range(num_episodes):
        hebbian_traces = agent_net.initialZeroHebb(1)
        hidden_activations = agent_net.initialZeroState(1)
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0)#.to(device) #This as well?
            policy_output, value, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards


                    

def evaluate_BP_agent_force_mag(agent_net, env_name, num_episodes, evaluation_seeds, force_mag_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.force_mag *= force_mag_modifier
        
    for i_episode in range(num_episodes):
        hebbian_traces = agent_net.initialZeroHebb(1)
        hidden_activations = agent_net.initialZeroState(1)
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0)#.to(device) #This as well?
            policy_output, value, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards



def evaluate_BP_agent_pole_mass(agent_net, env_name, num_episodes, evaluation_seeds, pole_mass_modifier):

    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.masspole *= pole_mass_modifier
        
    for i_episode in range(num_episodes):
        hebbian_traces = agent_net.initialZeroHebb(1)
        hidden_activations = agent_net.initialZeroState(1)
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0)#.to(device) #This as well?
            policy_output, value, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards



def evaluate_BP_agent_all_params(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier, pole_mass_modifier, force_mag_modifier):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)
        env.unwrapped.length *= pole_length_modifier
        env.unwrapped.masspole *= pole_mass_modifier
        env.unwrapped.force_mag *= force_mag_modifier
            
        for i_episode in range(num_episodes):
            hebbian_traces = agent_net.initialZeroHebb(1)
            hidden_activations = agent_net.initialZeroState(1)
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device) #This as well?
                
                policy_output, value, (hidden_activations, hebbian_traces) = agent_net.forward(state.float(), [hidden_activations, hebbian_traces])
                
                policy_dist = torch.softmax(policy_output, dim = 1)
                
                action = torch.argmax(policy_dist)
                

                state, r, done, _ = env.step(action.item())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards

def randomize_env_params(env, randomization_params):
    pole_length = env.unwrapped.length
    # gravity = env.unwrapped.gravity
    # masscart = env.unwrapped.masscart
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    params = [pole_length, masspole, force_mag]

    
    for i in range(len(params)):
        if isinstance(randomization_params[i], float):
            low = params[i] - params[i] * randomization_params[i]
            high = params[i] + params[i] * randomization_params[i]
        elif isinstance(randomization_params[i], tuple):
            low = params[i]*randomization_params[i][0]
            high = params[i]*randomization_params[i][1]
            
        params[i] = np.random.uniform(low, high)

    env.unwrapped.length = params[0]
    # env.unwrapped.gravity = params[1]
    # env.unwrapped.masscart = params[2]
    env.unwrapped.masspole = params[1]
    env.unwrapped.force_mag = params[2]

    return env
    # Based on version from BP paper:
    # def train_agent(self):
        

    #     all_losses = []
    #     all_grad_norms = []
    #     all_losses_objective = []
    #     all_total_rewards = []
    #     all_losses_v = []
    #     lossbetweensaves = 0
    #     nowtime = time.time()
        

    #     hidden = self.agent_net.initialZeroState(self.batch_size)
    #     hebb = self.agent_net.initialZeroHebb(self.batch_size)
        

    #     for i_episode in range(1, self.num_episodes + 1):
            
    #         self.optimizer.zero_grad()
    #         loss = 0
    #         lossv = 0
    #         self.hidden_activations = self.agent_net.initialZeroState(self.batch_size).to(device)
    #         self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size).to(device)
    #         numactionchosen = 0

    #         reward = np.zeros(self.batch_size)
    #         sumreward = np.zeros(self.batch_size)
    #         rewards = []
    #         vs = []
    #         logprobs = []
    #         dist = 0
    #         numactionschosen = np.zeros(self.batch_size, dtype='int32')
            
    #         state = self.env.reset()

    #         score = 0
    #         done = False
    #         while not done:
                
    #             output, value, (self.hidden_activations, self.hebbian_traces) = self.agent_net(state.float(), [self.hidden_activations, self.hebbian_traces])

    #             softmaxed_output = torch.softmax(output, dim = 1)
    #             distrib = torch.distributions.Categorical(softmaxed_output)
    #             selected_actions = distrib.sample()
    #             logprobs.append(distrib.log_prob(selected_actions))
    #             numactionschosen = selected_actions.cpu().numpy()

    #             next_state, reward, done, _ = self.env.step(numactionschosen)





    #         # self.optimizer.zero_grad()
    #         loss = 0
    #         lossv = 0
    #         self.hidden_activations = self.agent_net.initialZeroState(self.batch_size).to(device)
    #         self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size).to(device)
    #         numactionchosen = 0


            





    #         loss += ( self.entropy_coef * y.pow(2).sum() / self.batch_size )

    #         # # Episode is done, now let's do the actual computations of rewards and losses for the A2C algorithm

    #         # R is the sum of discounted rewards. It seems that 'rewards-to-go'
    #         # is implemented here, i.e. only the rewards received after a certain
    #         # time step are taken into account, because the action taken at that 
    #         # time step has no influence on the rewards before it.
    #         R = torch.zeros(self.batch_size).to(device)
    #         for numstepb in reversed(range(self.max_steps)) :
    #             R = self.gammaR * R + torch.from_numpy(rewards[numstepb]).to(device)
    #             ctrR = R - vs[numstepb][0]
    #             lossv += ctrR.pow(2).sum() / self.batch_size
    #             loss -= (logprobs[numstepb] * ctrR.detach()).sum() / self.batch_size  
    #             #pdb.set_trace()



    #         loss += self.value_pred_coef * lossv
    #         loss /= self.max_steps

    #         if PRINTTRACE:
    #             if True: #params['algo'] == 'A3C':
    #                 print("lossv: ", float(lossv))
    #             print ("Total reward for this episode (all):", sumreward, "Dist:", dist)

    #         loss.backward()
    #         all_grad_norms.append(torch.nn.utils.clip_grad_norm(self.agent_net.parameters(), self.max_grad_norm))
    #         if numiter > 100:  # Burn-in period for meanrewards #KEEP THIS?
    #             self.optimizer.step()


    #         lossnum = float(loss)
    #         lossbetweensaves += lossnum
    #         all_losses_objective.append(lossnum)
    #         all_total_rewards.append(sumreward.mean())
    #             #all_losses_v.append(lossv.data[0])
    #         #total_loss  += lossnum


    #         if (numiter+1) % self.print_every == 0:

    #             print(numiter, "====")
    #             print("Mean loss: ", lossbetweensaves / self.print_every)
    #             lossbetweensaves = 0
    #             print("Mean reward (across batch and last", self.print_every, "eps.): ", np.sum(all_total_rewards[-self.print_every:])/ self.print_every)
    #             #print("Mean reward (across batch): ", sumreward.mean())
    #             previoustime = nowtime
    #             nowtime = time.time()
    #             print("Time spent on last", self.print_every, "iters: ", nowtime - previoustime)
    #             #print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy())




    #         # More useful for when training takes longer
    #         # if (numiter+1) % self.save_every == 0:
    #         #     print("Saving files...")
    #         #     losslast100 = np.mean(all_losses_objective[-100:])
    #         #     print("Average loss over the last 100 episodes:", losslast100)
    #         #     print("Saving local files...")
    #         #     with open('grad_'+suffix+'.txt', 'w') as thefile:
    #         #         for item in all_grad_norms[::10]:
    #         #                 thefile.write("%s\n" % item)
    #         #     with open('loss_'+suffix+'.txt', 'w') as thefile:
    #         #         for item in all_total_rewards[::10]:
    #         #                 thefile.write("%s\n" % item)
    #         #     torch.save(self.agent_net.state_dict(), 'torchmodel_'+suffix+'.dat')
    #         #     with open('params_'+suffix+'.dat', 'wb') as fo:
    #         #         pickle.dump(params, fo)
    #         #     if os.path.isdir('/mnt/share/tmiconi'):
    #         #         print("Transferring to NFS storage...")
    #         #         for fn in ['params_'+suffix+'.dat', 'loss_'+suffix+'.txt', 'torchmodel_'+suffix+'.dat']:
    #         #             result = os.system(
    #         #                 'cp {} {}'.format(fn, '/mnt/share/tmiconi/modulmaze/'+fn))
    #         #         print("Done!")







