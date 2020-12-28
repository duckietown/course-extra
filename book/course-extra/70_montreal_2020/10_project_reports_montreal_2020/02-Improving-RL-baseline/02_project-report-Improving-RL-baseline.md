#  Improving the RL baseline: Project report {#improving-rl-baseline-final-report status=ready}

In this project, we propose to improve the Reinforcement Learning (RL) baseline [FIND HOW TO LINK TO THIS](https://github.com/duckietown/docs-AIDO/blob/master19/book/AIDO/31_task_embodied_strategies/36_rl_baseline.md)

## The final result {#improving-rl-baseline-final-result}

* VIDEO OF OUR AGENT - ideally video of episodes with the episode number displayed

You can find the [instructions to reproduce](#instructions-improving-rl-baseline) and code here LINK TO CODE.
## Mission and Scope {#improving-rl-baseline-final-scope}

The goal of this project is to improve the RL baseline in Duckietown by 


### Motivation {#improving-rl-baseline-final-result-motivation}

So far, two main approaches have been explored to control duckiebots : classical robotics methods and pure RL approaches.

- **Classical robotics methods** work well when the state information from the camera feed is correct and can ne interpreted in a meaningful way. However, they often require careful tuning of the parameters depending on the duckie / the environment (e.g. color range for line segments when the lighting conditions change) . 

- **RL approaches** allow the exploration of solutions that could not necessarily be found through classical methods or even imitating existing expert behavior. But they are often computationally expensive and data inefficient. In Duckietown, no pure RL approach seems to have outbeaten, or even matched the performance of classical methods.

[ADD SOMETHING] 


### Existing solution {#improving-rl-baseline-final-literature}

The current RL baseline consists of a Deep Deterministic Policy Gradient (DDPG) agent [](#bib:lillicrap2019continuous). 
In Duckietown, the action space is consists of the wheel commands $v$ and $\omega$, and is continuous. 
DDPG is an off-policy algorithm, in which both an approximator to the optimal Q function and an approximator to the optimal action function are learnt. It can be thought of as Deep Q-learning [PUT A REFERENCE] for continuous action spaces.

[ELABORATE?]

Learn more about DDPG [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).



### Opportunity {#improving-rl-baseline-final-opportunity}

The RL approaches explored so far were pure RL approaches.

After a day of training on a Nvidia RTX2060 mobile paired with an Intel i7-9750H and 16 GB of RAM, the agent of the RL baseline was going forward turning to the right straight out of lane independently of the configuration of the lane, and was still far from following the lane. 

Limitations of such pure RL models might include the lack of computing power since it is not possible to take advantage of the hardware acceleration on acomputation cluster, or bad parameters initialization or reward function that didn’t foster convergence to a competent agent.

Moreover, the agent is trained in the Duckietown simulator, and there is no guarantee that transferring from sim to real will be successful with this approach.

#### Contribution{#improving-rl-baseline-final-opportunity-contribution}

We propose to train a DARLA agent [](#bib:higgins2018darla).  
There are three steps to follow: 
- Learn to see: solve the perception task. The goal is to learn a disentangled representation of the environment to be robust to domain shifts.
- Learn to act: train an RL agent
- Transfer: evaluate on new target domain without retraining.

This approach is particularly interesting for Duckietown because of the domain shifts due to the variation of parameters in the simulator or the sim2real gap. Indeed, Higgins et al. argue that if a good disentangled representation of the environment is learned, the model can be transferred to new domains without further training. 
Instead of feeding the camera images to the RL model, we project the agent observation state space  to a latent state space expressed in terms of factorised data generative factors that are representative of the natural world, and use this latent representation as the input.
<this is too wordy>

## Background and Preliminaries {#improving-rl-baseline-final-preliminaries}

* Is there some particular theorem / "mathy" thing you require your readers to know before delving in the actual problem? Briefly explain it and links for more detailed explanations here.

Definition of link:
- could be the reference to a paper / textbook
- (bonus points) it is best if it is a link to Duckiebook chapter (in the dedicated "Preliminaries" section)

## Definition of the problem {#improving-rl-baseline-final-problem-def}

_Up to now it was all fun and giggles. This is the most important part of your report: a crisp, possibly mathematical, definition of the problem you tackled. You can use part of the preliminary design document to fill this section._

Make sure you include your:
- final objective / goal
- assumptions made
- quantitative performance metrics to judge the achievement of the goal

## Contribution / Added functionality {#improving-rl-baseline-final-contribution}

Describe here, in technical detail, what you have done. Make sure you include:
- a theoretical description of the algorithm(s) you implemented
- logical architecture
- software architecture
- details on the actual implementation where relevant (how does the implementation differ from the theory?)
- any infrastructure you had to develop in order to implement your algorithm
- If you have collected a number of logs, add link to where you stored them

_Feel free to create subsections when useful to ease the flow_

### Model architecture
[DRAWING OF THE ARCHITECTURE]
### Dataset 
[LINK TO DATASET ? or code to collect data] 
### DAE 
We first train a DAE to 
### Beta VAE
### RL agent 

## Formal performance evaluation / Results {#improving-rl-baseline-final-formal}

### Denoising Auto Encoder
We train the DAE on our simulated images dataset for XXX epochs. 
something about the loss
[IMAGE of some RECONSTRUCTIONS]
_Be rigorous!_

### Beta Variational Auto Encoder

[IMAGE of traversals]

- For each of the tasks you defined in you problem formulation, provide quantitative results (i.e., the evaluation of the previously introduced performance metrics)
- Compare your results to the success targets. Explain successes or failures.
- Compare your results to the "state of the art" / previous implementation where relevant. Explain failure / success.
- Include an explanation / discussion of the results. Where things (as / better than / worst than) you expected? What were the biggest challenges?

## Future avenues of development {#improving-rl-baseline-final-next-steps}

_Is there something you think still needs to be done or could be improved? List it here, and be specific!_

&lt;div id="put-bibliography-here"&gt;&lt;/div&gt;