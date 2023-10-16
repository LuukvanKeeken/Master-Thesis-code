from copy import deepcopy
import torch
import numpy as np
import gym
import random
import time
from collections import deque


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class A2C_Agent:

    def __init__(self, env_name, seed, agent_net, entropy_coef, value_pred_coef, gammaR, max_grad_norm, max_steps, batch_size,
                 num_training_episodes, optimizer, print_every, save_every, i_run, result_dir, best_model_selection_method,
                 num_evaluation_episodes, evaluation_seeds, max_reward):
        
        if batch_size > 1:
            print("Batch size larger than 1 not implemented yet. Program will continue with batch size set to 1.")
            

        self.env = gym.make(env_name)
        self.env.seed(seed)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.n

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
        self.print_every = print_every # number of episodes between printing the current average score
        self.save_every = save_every # number of episodes between saving the most recent model
        self.i_run = i_run
        self.result_dir = result_dir
        self.selection_method = best_model_selection_method
        self.num_evaluation_episodes = num_evaluation_episodes
        self.evaluation_seeds = evaluation_seeds
        self.max_reward = max_reward


        # Initialize Hebbian traces
        self.hebbian_traces = self.agent_net.initialZeroHebb(self.batch_size).to(device)

        # Initialize hidden activations
        self.hidden_activations = self.agent_net.initialZeroState(self.batch_size).to(device)


    def train_agent(self):

        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen = 100)


        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        

        for episode in range(1, self.num_training_episodes + 1):
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

                    if ((self.selection_method == "evaluation") and (episode % 10 == 0)):
                        evaluation_performance = np.mean(evaluate_BP_agent(self.agent_net, self.env_name, self.num_evaluation_episodes, self.evaluation_seeds, 1.0))
                        print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                        if evaluation_performance > best_average:
                            best_average = evaluation_performance
                            best_average_after = episode
                            torch.save(self.agent_net.state_dict(),
                                       self.result_dir + '/checkpoint_BP_A2C_{}.pt'.format(self.i_run))
                            

    
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
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

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
        eval_rewards = evaluate_BP_agent(self.agent_net, self.env_name, n_evaluations, evaluation_seeds, pole_length_modifier)
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
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()


            if (i_episode % eval_skip == 0):
                eval_rewards = evaluate_BP_agent(self.agent_net, self.env_name, n_evaluations, evaluation_seeds, pole_length_modifier)
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


def evaluate_BP_agent(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):

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







