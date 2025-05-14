# MAGS: Dynamic Memory Modules for Large Language Models via Liquid Knowledge Graphs

## Technical Specifications
### Extended Abstract

Currently, Large Language Models (LLMs), while seemingly intelligent, lack one critical component of intelligence–the ability to learn. While LLM’s can emulate learning at test-time with the context window, this approach is likely not scalable nor robust, largely because the processing power needed to maintain and attend to these windows scale quadratically with its length. Additionally, with large context windows models struggle with long-term retention, as they do not internalize knowledge but rather treat all context as ephemeral. Furthermore, there is attention decay, where LLMs will struggle to focus on early context (and ‘forget’ it), and LLMs cannot retain this information between sessions without recreating the entire chat.

We propose a neuroscience-inspired architecture to address these shortcomings and provide LLMs with knowledge blocks more akin to human memory. This design incorporates Liquid Knowledge Graphs (LKGs) that can dynamically adapt at test-time to provide models with the ability to learn continuously.

Knowledge Graphs have been widely incorporated into LLMs before (typically in the form of Retrieval Augmented Generation, or RAG), but they are typically used as static sources of external grounding rather than as dynamic memory blocks. They rely on user-controlled intervention for updates, and LLMs cannot directly modify or store information in them—resulting in a one-way transfer of knowledge from the graph to the model. Additionally, they have no method for reconciling contradicting information. 

We introduce MAGS (Memory Augmented Generative System), an architecture involving two memory blocks (MBs) that together emulate episodic (eMB) and semantic memory (sMB). The semantic memory replicates the persistent (long-term) parametric memory LLMs are fixed to, but allow for updates and changes (liquidity) without the need for retraining.

Training and Specifics:
Reinforcement Learning (RL) via Group Relative Policy Optimization (GRPO), along with Low Rank Adaptation (LoRA), will be used to guide and fine-tune the model to maximally utilize these memory blocks. Tokens, similar to <thinking> tokens using by reasoning models, will be encouraged for the model to recall (fetch) information (the recall step). Each model response will be accompanied by requests for additions or changes to memory (the engram step). Contradicting memory will be addressed via KEDKG, and a ‘The Game of Life’-esque algorithm will be used to strengthen or weaken edge weights, with a recency correction to retain knowledge used recently. A pruning algorithm for edges with low weights will be employed to keep the LGKs from becoming overly dense and exceeding environment memory constraints, while exceptionally strong edge weights might be permanently solidified. We will use k-core decomposition to retain and consolidate central nodes.

We must also know how to manage (add and remove) nodes, and add edges. The engram step will take new information, and add new nodes for information that was not previously captured (using similarity). Otherwise, it will modify nodes for information that was largely already captured. When a new node is added, inspiration and techniques from Graph Attention Networks (GATs) will be used to autonomously add initial connections. Isolated nodes or inactive nodes (nodes that have not been used for a large period of time or have had low degrees for extended periods of time) will be removed along with all outgoing connections. 

Adding new connections will likely be the most complicated. We want to balance the two main methods the human brain uses, which are associative learning (Hebbian Plasticity–’neurons that fire together, wire together’) and Neurogenesis / Competitive Plasticity. Neurogensis follows the addition of new neurons (nodes), and was address above. Thus, we must replicate Hebbian Plasticity. 

During the recall process, nodes with the best similarity to the prompt (the RL process hopefully will encourage processes like multiquery and step-back to better fetch relevant information) will be chosen as candidate ‘anchored’ nodes. Initially, we will over-query for anchor nodes, and choose a constant number of these candidate anchors (to be actually anchors) whose convex hull maximize captured graph area. These anchor nodes will then be connected to emulate Hebbian Plasticity. Then a Depth-First-Search will be performed from each anchor node, until the edge of the graph is reached, or the product of edge weights of the path falls below a specified threshold. Logical routing will be performed to balance recall from both the eMB and sMB. Thus, this edge weighting is vital. During the engram step, for paths that were more impactful in the final response, edges along that path will be strengthened (this replicates reward-based learning, which is where connections that led to successful outcomes are strengthened).

As seen with thinking modes, this RL sometimes leads the model to ‘think’ and work in its own language or in its own unexpected way. We will attach guardrails as described in the DeepSeek paper to force the model to stay in English and our specified structure. This ensures we can correctly parse which nodes and connections to update, as well as explainability purposes.

This initial Chains-of-Memory (CoMs, from the recall and engram steps) will likely have to initially be developed by hand, and then few-shotted with a thinking mode to develop more examples (and reviewed for accuracy, of course). 

Testing:
To test MAGS, we need to develop a novel game, and provide the rules to the model. We should use a non-thinking model (i.e. typical flash models) so that as MAGS plays the game it can store interesting or unique findings, without discovering those findings through the game with logic. This way, we can better evaluate the knowledge it's storing and how it's using it. As the model plays the game, we will introduce new rules or behaviors and test its adaptation. Testing will be done with multi-shot prompts, with feedback to evolve memory. 

Because memory is stored as a separable entity, we can directly modify this memory, as well as inspect it for explainability. This would allow for robustness and correctness testing as well. This also allows change to memory blocks will be rolled back between game instances, since we can keep snapshots.



### Possible Avenues of Exploration
#### Possible exploration paths (not fully thought out):
- We want to also explore how thinking can connect with this memory model. Perhaps thinking steps can fetch further memory elements to better guide its process. Additionally, it is likely connections and similarities made during the thinking stage should be added to the memory blocks as well. 
- Removing context entirely and specifying the user to be a ‘manager’. All important context is then stored in the eMB as conversations with this ‘manager’, allowing a ‘fresh’ LLM to be used for every prompt. 
- Importance Scores for nodes (can be a function of degree weights times outdegree perhaps). Emulates Salience-Based - Connections of human brains, where events that were emotionally intense, surprising, or critical form stronger and faster connections.
- How we are planning to compartmentalize the information. We want to create nodes that are future-proof, so want nodes to be general, how how can we do this?

#### Smaller Considerations and possible additions:
- Thematic clustering of information.
- Confidence scores appointed to nodes for how correct they think the information is. Can likely help with reconciling contradicting information.
- Recency scores for nodes as well to be used in tiebreakers.
- Somehow tying together eMB and sMB?? Have no clue tbh. Perhaps using something like entity linking.


# Memory Augmented Generative System (MAGS): A Detailed Research Proposal

This document outlines our research plan for MAGS, a neuroscience-inspired architecture that endows large language models (LLMs) with the ability to continuously learn via dynamic memory blocks. Our approach directly addresses the limitations of current LLMs, which rely on ephemeral context windows that scale poorly, suffer from attention decay, and cannot retain information between sessions.

---

## Problem Statement

### What Are We Optimizing?

We are optimizing the integration of a continuous, test-time learning mechanism into LLMs by:
- **Enhancing Memory Recall:** Designing dual memory blocks—a short-term Episodic Memory Block (eMB) and a long-term Semantic Memory Block (sMB)—to retrieve and integrate context efficiently.
- **Improving Memory Updates:** Developing a system that allows on-the-fly updates to the memory structure through reinforcement learning (using Group Relative Policy Optimization, GRPO) and Low Rank Adaptation (LoRA), ensuring that the memory reflects new information without retraining the entire model.

### Why Does This Problem Matter?

Current LLMs simulate learning only by extending the context window, which:
- **Scales Poorly:** The quadratic increase in computational requirements with window length makes it inefficient.
- **Lacks Persistence:** Models treat all context as ephemeral, leading to rapid attention decay and loss of early session information.
- **Hinders Adaptability:** LLMs cannot reconcile contradicting information or update their stored knowledge without a full retraining cycle.

By implementing MAGS, we aim to:
- **Mimic Human Memory:** Emulate episodic (short-term) and semantic (long-term) memory functions, similar to how humans store and recall information.
- **Enable Continuous Learning:** Allow models to adapt dynamically in changing environments (e.g., evolving game rules) without expensive retraining.
- **Ensure Explainability and Robustness:** Maintain a separable, inspectable memory structure that can be modified and rolled back if needed.

### How Will We Measure Success?

Using a novel game, success will be evaluated using the following metrics:
- **Recall Accuracy:** The percentage of questions answered correcty using information from the recall step.
- **Update Fidelity:** The consistency and correctness of memory updates during the engram step, verified against controlled ground truth data.
- **Adaptability:** Improvement in task performance (e.g., in a novel game environment where rules are periodically changed).
- **Computational Efficiency:** Effective management of memory graph density through pruning and k-core decomposition, ensuring real-time performance.
- **Explainability:** Ability to directly inspect and manually adjust memory graphs for debugging and analysis.

### Constraints and Data Requirements



- **Computational Requirements:** Access to a 16-24GB VRAM GPU (e.g. RTX 3090), likely through cloud computing / credits. 
- **Data Needs:** Synthetic datasets mimicking episodic events, controlled game simulations with evolving rules, and logs of memory updates for rigorous testing. We will liekly generate a few samples ourselves, and then use generative tools to make more samples, similar to how DeepSeek got their initial dataset.
- **Major Constraints:** Our constraints are largely the lack of compute, and the lack of a dataset to train on (mentioned above). There is also little past research on this topic, so we will need to guide ourselves mostly.


### Potential Failures
- **Potential Failures:** Risks include overgrowth of memory graphs, inconsistent memory updates, slow RL convergence, and memory contradictions that degrade model performance. There is also just the overall risk that this method does not provide any addiitonal benifit 
over typical inference methods, or we do not test it in a way to observe these benifits.
---

## Technical Approach

### Mathematical Formulation

We formalize the optimization as follows:

\[
\min_{\theta, \phi} \; \mathcal{L} = \mathcal{L}_{\text{LLM}}(\theta, \phi) - \lambda \, \mathbb{E}\bigl[R_{\text{MB}}\bigr] + \mu \, \mathcal{L}_{\text{graph}}
\]

- **\(\mathcal{L}_{\text{LLM}}(\theta, \phi)\):** The standard language modeling loss over parameters \(\theta\) (LLM weights) and \(\phi\) (parameters governing memory integration).
- **\(R_{\text{MB}}\):** A reward signal that quantifies the success of memory recall and engram updates. This is optimized using GRPO.
- **\(\mathcal{L}_{\text{graph}}\):** A penalty term for inefficient graph structures (e.g., overly dense nodes, low-weight edges), mitigated via a “Game of Life”-inspired algorithm and k-core decomposition.
- **Constraints:** Include limits on available memory and computational time, ensuring the update mechanism remains efficient (sub-quadratic in complexity).

### Algorithm and Implementation Choice

- **Dual Memory Architecture:**
  - **Episodic Memory Block (eMB):** Captures short-term, session-specific context.
  - **Semantic Memory Block (sMB):** Maintains long-term, persistent knowledge, updated continuously using LoRA without retraining the full model.
  
- **Memory Update Process:**
  - **Recall Step:**
    - Identify candidate anchor nodes based on similarity to the current prompt.
    - Over-query for anchors and select a fixed number whose convex hull maximizes graph area coverage.
    - Perform a depth-first search (DFS) from these anchors, terminating paths when the product of edge weights falls below a threshold.
  - **Engram Step:**
    - Integrate new information by either creating new nodes (using similarity measures and GAT-inspired initial edge connections) or updating existing nodes.
    - Strengthen edges along paths that contribute positively to the response (Hebbian plasticity), while applying recency corrections and pruning low-weight or inactive nodes.
    - Utilize KEDKG to reconcile conflicting information.

- **PyTorch Implementation Strategy:**
  - **Modular Design:** Develop separate PyTorch modules for eMB and sMB, interfacing seamlessly with the LLM.
  - **Graph Operations:** Employ PyTorch Geometric to handle dynamic graph creation, node similarity computations, and edge updates.
  - **Reinforcement Learning Module:** Implement GRPO for guiding both recall and engram steps, ensuring that successful memory updates receive positive reinforcement.
  - **Guardrails:** Integrate constraints (as per the DeepSeek paper) to enforce output in English and adhere to a specified node/edge structure for easier parsing and debugging.

### Validation Methods and Resource Requirements

- **Validation:**
  - **Synthetic Gameplay Testing:** Develop a controlled game environment where the model is exposed to evolving rules. Measure adaptability and retention as the model plays.
  - **Ablation Studies:** Isolate and test individual components (e.g., Hebbian updates, DFS recall, pruning algorithms) to assess their contributions.
  - **Explainability Audits:** Regularly inspect memory graphs to ensure modifications align with expectations, and test the ability to roll back memory changes.
  
- **Resource Requirements:**
  - **Hardware:** High-performance GPU clusters to manage dynamic graph computations and RL loops.
  - **Memory:** Sufficient RAM/VRAM to support large-scale dynamic graphs and fast retrieval.
  - **Computational Budget:** Ensure the implementation remains efficient by controlling graph density and update frequency to avoid performance bottlenecks.

---

## Initial Results

### Working Implementation
- Our repo has the code for a basic implementation of a RAG database, with the ability to query from it. You can add and remove nodes, and ask questions based on this. Right now it's not much more than a very basic Graph RAG.
- We implemented dataloaders and a custom game to evaluate / test on, although we have no official metrics. We do have a working implementation however.


### Performance Metrics
- We don't really have any performance metrics as of it. We can see that when we provide the model with new information it typically adds 1-2 nodes, although it seems very nondeterminsitic / not robust.
- It also seems to be taking about ~1min per response, which is likely do to our output format, so we might need to optimize (since its local).
### Test Cases
- Its hard to right test cases for this since its a LLM with no right answer, but our graph-test cases pass meaning that our graph library is strong (albeit sparse)
### Limitations
- We haven't implemented much of the MAGS functionality as of it. The LLM can add to the graph, but right now its very simplistic.
- It also often adds random or wrong infrormation that doesn't entirely pertain to our prompt.
- It also takes a long time to respond, even to simple question.
### Resource Usage
- Using CPU, doesn't seem like a high utilization (< 20%). No GPU as of yet.
- Again, slow responses.
### Unexpected Challenges
- Trying to autoload to models from hugging face hit some rate limit and I got banned lol
- We realize how, even on temperature 0, its giving nonderterminstic models, which it shouldn't be? Hard to evaluate when its doing so.


## Next Steps

### Immediate Improvements
- Start to implement actual RAG, with decay and strengthening and graph algorithms
- Implementent achors / traversals rather thna RAW RAG
- Fine tune an actual model.
### Technical Challenges to Address
- Get some GPU, either Colab or Metal GPU
- Figure out how to fine-tune an LLM (i.e. LoRA or PPO or GPO)
- Create the datasets
### Questions for Help
- Honestly, as of now, we think we can handle it, no questions as of yet!
### Alternative Approaches to Try
- Since we don't have any major roadblocks, we don't really have any alternative approaches here. We have a lot of ideas which likely will need some in the future however! Our main issue is GPU compute, so our approach as of now is to use smaller paramater models.
### What we've learned so far
- We learned a lot about running and using local models. We largely already knew a lot about RAG so much of this stuff was repeating stuff we've done in the past, but it was stil really cool to see it all work. 
- We've done a lot of literature review in the past couple of weeks, so we've learned a LOT about current current approaches for long-term memory or context, like RoPE, and Titans!
