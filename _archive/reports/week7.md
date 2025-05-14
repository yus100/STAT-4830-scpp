# MAGS: Dynamic Memory Modules for Large Language Models via Liquid Knowledge Graphs




## Problem Statement

The problem remains developing an architecture that enables long-term model learning.

### What Are We Optimizing?

No Change.



### Why Does This Problem Matter?

No Change.



### How Will We Measure Success?

No Change.


### Constraints / Data Requirements

No Change.


### Potential Failures
No Change.

---

## Technical Approach

### Mathematical Formulation

We updated our mathematical formula to be two parts. The first part is a loss metric for the model, and the second is a non-differentiable loss function for the graph to penalize multiple things.

Let $\pi_\phi(a \mid s)$ be the policy with parameters $\phi$ and $\pi_{\phi_{\text{old}}}(a \mid s)$ be the policy before the update. Define the probability ratio:

The first function is the loss metric for the model. This is the generic GRPO loss metric, although we liely will use PPO for our intial model since its more stable, easier to debug, and we are more familiar with it. Our end goal is to use GRPO because of its higher performance.

$r_t(\phi) = \frac{\pi_\phi(a_t \mid s_t)}{\pi_{\phi_{\text{old}}}(a_t \mid s_t)}$

Let $A_t$ be the advantage estimate at time $t$ and define the group-relative advantage as:

$\hat{A}t^{\text{GRPO}} = A_t - \frac{1}{|\mathcal{G}(t)|} \sum{t’ \in \mathcal{G}(t)} A_{t’}$

where $\mathcal{G}(t)$ is the set of experiences in the group corresponding to time $t$.

Then the GRPO objective is:

$L^{\text{GRPO}}(\phi) = \mathbb{E}_t \left[ \min \left( r_t(\phi) \hat{A}_t^{\text{GRPO}}, \ \text{clip} \left( r_t(\phi), 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t^{\text{GRPO}} \right) \right]$

where $\epsilon$ is a hyperparameter that limits the extent of policy updates.

The second function is a 4-term non-differentiable loss function for the graph to penalize multiple things.

$\min_{\theta, \phi, \psi} \mathcal{L} = 
\mathbb{E}{(x, y) \sim \mathcal{D}} \left[ \ell\left(f\theta\left(x, g_\phi\left(M(x; \psi)\right)\right), y \right) \right]$

$ + \lambda \, \mathbb{E}{x \sim \mathcal{D}} \left[ R\left(M(x; \psi), g\phi\left(M(x; \psi)\right)\right) \right]$

$+\mu \, \mathcal{L}_{\text{graph}} \left(M(x; \psi)\right)$

$ + \nu \, \mathcal{R}(\theta, \phi, \psi)$

Our total loss function balances prediction accuracy with graph efficiency.

Four terms:
	•	Prediction Loss: Accuracy of model responses on the evaluation / test set
	•	Graph Regularization: Term that penalizes excessive recalls / engrams per query
	•	General Regularization: Term that penalizes excessive nodes / edges in terms of memory
	•	Memory Recall Loss: Term that penalizes rapid changes to the graph, to prevent constant graph liqudity / costly changes constantly.

⸻

We define our objective as a composite loss that jointly optimizes the language model’s prediction accuracy, memory retrieval/reinforcement, and dynamic graph structure regularization.

Let:
	•	$\theta$ denote the parameters of the underlying LLM.
	•	$\phi$ denote the parameters governing the memory retrieval and update module.
	•	$\psi$ denote the parameters controlling the dynamic graph (i.e., Liquid Knowledge Graph) structure.
	•	$\mathcal{D} = {(x, y)}$ be the dataset of input–output pairs.
	•	$M(x; \psi)$ be the memory representation extracted from input $x$ (including both episodic and semantic components).
	•	$g_\phi\left(M(x; \psi)\right)$ be the memory retrieval function that selects relevant memory nodes.
	•	$f_\theta(\cdot)$ be the generative function of the LLM augmented with the retrieved memory.
	•	$\ell(\cdot, \cdot)$ be a standard prediction loss (e.g., cross-entropy).


⸻




### Algorithm and Implementation Choice

We have decided to change to two models (one for engram and one for recall). Trying to combine them and fine-tune seems to work, butn when evaluating it, we get no homogenity, and it seems to start spewing random combinations of information. We will probably create a router that sends a query to the right model, and have each model specialize. 

### Validation Methods and Resource Requirements

Validation is the same. We are using small models (1B-1.5B), mostly Llama, since we don't have access to higher performance clusters to run the models, and are during locally. We can run small models on our laptop, although it will take time to train and evaluation. For the initial stages, however, this should be fine.

---

## Initial Results

### Working Implementation
We mostly spent this week working on training the model with PPO. We have the working code in `src/RL/PPO.py`, and running the python script. We have our Eval script in `src/RL/Eval.py`. We also have some older versions called `MemoryAgent.py` although this one didn't work nearly as well. The model trains well (although very slow)


### Performance Metrics

Performance with PPO is decent but not great. We're going to try with a few models (so far we just did LLama 1B, but want to expand to some larger models). We didn't really find good accuracy, it was at about 30% F1 score right now (a combination of precision+recall, and a raw score for getting the right function as well as homogenity). We haven't really had time to evaluate this with the main graph system yet.

### Test Cases

As of now, we didn't have any test cases, since we didn't integrate it, and its not really possivble to write tes cases for an LLM for this use case it seems like.

### Limitations

Main limitaitons right now is that this code is extremely slow and not very accurate. It took around 31 hours to train on a macbook (CPU, no GPU acceleration)

Also not accurate, aboutn 30% and often just spews random garbage.

### Resource Usage

Uses almost all of the CPU(90-100%) on my work laptop, and heats up very much, but only uses a few gigs of memory. Seems like speed is the bottleneck here or my code just isn't very good lol.

### Unexpected Challenges

Didn't expect it to take so long, I thought PPO was supposed to take ~8 hours for 1K samples on a macbook, so a bit confused. My code is likely just not effecient.

Also didn't expect the accuracy to be so bad, seems like the model is not learning the difference between engram and recall vey well.

Multishot prompting to make the datasets had almost no variability, often just repeats of the same example, will likely need to make a better prompt or a better example set.

## Next Steps

### Immediate Improvements

- Seperating the engram and recall steps into seperate models and creating new routing model (likely just GPT or 1.5B local model since its simple)

- Integrating this model with the graph to see its performance.

- Trying out larger models and expanding the dataset.

### Technical Challenges to Address

- Need to figre out why PPO is so slow and otpimize it
- Need to figure out how to prompt for a better synthetic dataset
- Need to figure out the architecutre for the router to not slow down the model too much.
### Questions for Help
 - Main question is why my PyTorch code is so slow (although you'll probably have to spend a lot of time looking at the code since its not very well organized lol.)
### Alternative Approaches to Try
- As mentioned, using a routing model and 2 small models instead of one fine-tuned model to help with seperatng engram and recall
- Using larger models to see if this helps the adherence to the prompt
### What we've learned so far
- Learned a lot about how PPO works and implementing it from scratch! Learned a lot on how to actually write PyTorch programs (my first one!). Used a lot of GPT for this, but still learned all the fundementals. Also learned how slow fine-tuning can be :(.
