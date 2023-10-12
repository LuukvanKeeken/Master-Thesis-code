import torch
import numpy as np
import gym
import random
import time


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class A2C_Agent:

    def __init__(self, env, seed, agent_net, entropy_coef, value_pred_coef, gammaR, max_grad_norm, max_steps, batch_size,
                 num_episodes, optimizer, print_every, save_every):
        self.env = gym.make(env)
        self.env.seed(seed)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.agent_net = agent_net
        self.entropy_coef = entropy_coef # coefficient for the entropy reward (really Simpson index concentration measure)
        self.value_pred_coef = value_pred_coef # coefficient for value prediction loss
        self.gammaR = gammaR # discounting factor for rewards
        self.max_grad_norm = max_grad_norm # maximum gradient norm, used in gradient clipping
        self.max_steps = max_steps # maximum length of an episode
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.optimizer = optimizer
        self.print_every = print_every # number of episodes between printing the current average score
        self.save_every = save_every # number of episodes between saving the most recent model










    def train_agent(self):
        

        all_losses = []
        all_grad_norms = []
        all_losses_objective = []
        all_total_rewards = []
        all_losses_v = []
        lossbetweensaves = 0
        nowtime = time.time()
        

        hidden = self.agent_net.initialZeroState(self.batch_size)
        hebb = self.agent_net.initialZeroHebb(self.batch_size)
        

        for numiter in range(self.num_episodes):
            
            self.optimizer.zero_grad()
            loss = 0
            lossv = 0
            hidden = self.agent_net.initialZeroState(self.batch_size).to(device)
            hebb = self.agent_net.initialZeroHebb(self.batch_size).to(device)
            numactionchosen = 0


            reward = np.zeros(self.batch_size)
            sumreward = np.zeros(self.batch_size)
            rewards = []
            vs = []
            logprobs = []
            dist = 0
            numactionschosen = np.zeros(self.batch_size, dtype='int32')

            #reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)

            #print("EPISODE ", numiter)
            for numstep in range(self.max_steps):



                inputs = np.zeros((self.batch_size, TOTALNBINPUTS), dtype='float32') 
            
                labg = lab.copy()
                for nb in range(self.batch_size):
                    inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE//2:posr[nb] + RFSIZE//2 +1, posc[nb] - RFSIZE //2:posc[nb] + RFSIZE//2 +1].flatten() * 1.0
                    
                    # Previous chosen action
                    inputs[nb, RFSIZE * RFSIZE +1] = 1.0 # Bias neuron
                    inputs[nb, RFSIZE * RFSIZE +2] = numstep / self.max_steps
                    inputs[nb, RFSIZE * RFSIZE +3] = 1.0 * reward[nb]
                    inputs[nb, RFSIZE * RFSIZE + ADDITIONALINPUTS + numactionschosen[nb]] = 1
                
                inputsC = torch.from_numpy(inputs).to(device)

                ## Running the network
                y, v, (hidden, hebb) = self.agent_net(inputsC, (hidden, hebb))  # y  should output raw scores, not probas


                y = torch.softmax(y, dim=1)
                distrib = torch.distributions.Categorical(y)
                actionschosen = distrib.sample()  
                logprobs.append(distrib.log_prob(actionschosen))
                numactionschosen = actionschosen.data.cpu().numpy()  # We want to break gradients
                reward = np.zeros(self.batch_size, dtype='float32')


                for nb in range(self.batch_size):
                    myreward = 0
                    numactionchosen = numactionschosen[nb]

                    tgtposc = posc[nb]
                    tgtposr = posr[nb]
                    if numactionchosen == 0:  # Up
                        tgtposr -= 1
                    elif numactionchosen == 1:  # Down
                        tgtposr += 1
                    elif numactionchosen == 2:  # Left
                        tgtposc -= 1
                    elif numactionchosen == 3:  # Right
                        tgtposc += 1
                    else:
                        raise ValueError("Wrong Action")
                    
                    reward[nb] = 0.0  # The reward for this step
                    if lab[tgtposr][tgtposc] == 1:
                        reward[nb] -= params['wp']
                    else:
                        posc[nb] = tgtposc
                        posr[nb] = tgtposr

                    # Did we hit the reward location ? Increase reward and teleport!
                    # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move...
                    # But we still avoid it.
                    if rposr[nb] == posr[nb] and rposc[nb] == posc[nb]:
                        reward[nb] += params['rew']
                        posr[nb]= np.random.randint(1, LABSIZE - 1)
                        posc[nb] = np.random.randint(1, LABSIZE - 1)
                        while lab[posr[nb], posc[nb]] == 1 or (rposr[nb] == posr[nb] and rposc[nb] == posc[nb]):
                            posr[nb] = np.random.randint(1, LABSIZE - 1)
                            posc[nb] = np.random.randint(1, LABSIZE - 1)

                rewards.append(reward)
                vs.append(v)
                sumreward += reward

                # This is an "entropy penalty", implemented by the sum-of-squares of the probabilities because our version of PyTorch did not have an entropy() function.
                # The result is the same: to penalize concentration, i.e. encourage diversity in chosen actions.
                loss += ( self.entropy_coef * y.pow(2).sum() / self.batch_size )  


                #if PRINTTRACE:
                #    print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
                #            #" - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), 
                #            " -Reward (this step, 1st in batch): ", reward[0])



            # Episode is done, now let's do the actual computations of rewards and losses for the A2C algorithm


            R = torch.zeros(self.batch_size).to(device)
            for numstepb in reversed(range(self.max_steps)) :
                R = self.gammaR * R + torch.from_numpy(rewards[numstepb]).to(device)
                ctrR = R - vs[numstepb][0]
                lossv += ctrR.pow(2).sum() / self.batch_size
                loss -= (logprobs[numstepb] * ctrR.detach()).sum() / self.batch_size  
                #pdb.set_trace()



            loss += self.value_pred_coef * lossv
            loss /= self.max_steps

            if PRINTTRACE:
                if True: #params['algo'] == 'A3C':
                    print("lossv: ", float(lossv))
                print ("Total reward for this episode (all):", sumreward, "Dist:", dist)

            loss.backward()
            all_grad_norms.append(torch.nn.utils.clip_grad_norm(self.agent_net.parameters(), self.max_grad_norm))
            if numiter > 100:  # Burn-in period for meanrewards
                self.optimizer.step()


            lossnum = float(loss)
            lossbetweensaves += lossnum
            all_losses_objective.append(lossnum)
            all_total_rewards.append(sumreward.mean())
                #all_losses_v.append(lossv.data[0])
            #total_loss  += lossnum


            if (numiter+1) % self.print_every == 0:

                print(numiter, "====")
                print("Mean loss: ", lossbetweensaves / self.print_every)
                lossbetweensaves = 0
                print("Mean reward (across batch and last", self.print_every, "eps.): ", np.sum(all_total_rewards[-self.print_every:])/ self.print_every)
                #print("Mean reward (across batch): ", sumreward.mean())
                previoustime = nowtime
                nowtime = time.time()
                print("Time spent on last", self.print_every, "iters: ", nowtime - previoustime)
                #print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy())




            # More useful for when training takes longer
            # if (numiter+1) % self.save_every == 0:
            #     print("Saving files...")
            #     losslast100 = np.mean(all_losses_objective[-100:])
            #     print("Average loss over the last 100 episodes:", losslast100)
            #     print("Saving local files...")
            #     with open('grad_'+suffix+'.txt', 'w') as thefile:
            #         for item in all_grad_norms[::10]:
            #                 thefile.write("%s\n" % item)
            #     with open('loss_'+suffix+'.txt', 'w') as thefile:
            #         for item in all_total_rewards[::10]:
            #                 thefile.write("%s\n" % item)
            #     torch.save(self.agent_net.state_dict(), 'torchmodel_'+suffix+'.dat')
            #     with open('params_'+suffix+'.dat', 'wb') as fo:
            #         pickle.dump(params, fo)
            #     if os.path.isdir('/mnt/share/tmiconi'):
            #         print("Transferring to NFS storage...")
            #         for fn in ['params_'+suffix+'.dat', 'loss_'+suffix+'.txt', 'torchmodel_'+suffix+'.dat']:
            #             result = os.system(
            #                 'cp {} {}'.format(fn, '/mnt/share/tmiconi/modulmaze/'+fn))
            #         print("Done!")







