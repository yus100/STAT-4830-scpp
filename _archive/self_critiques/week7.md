# Self-Critique: MAGS - Dynamic Memory Modules for Large Language Models via Liquid Knowledge Graphs

## OBSERVE
- The core problem statement remains consistent, and the high-level vision is the same.
- The mathematical formulation has evolved -- a distinction is now made between PPO/GRPO and a separate non-differentiable graph regularization loss. The specific terms for the graph loss are listed conceptually, and how this is non-differentiable will be incorporated into the optimization process is not linked to implementation.
- Decision to pivot from single fine-tuned model controlling both recall and engram steps to separate specialized models for each function, plus routing mechanism is a result of poor performance noted earlier.
- Focused on implementing and training a PPO agent intended to generate the recall/engram actions. Code for evaluation also exists.
- Initial PPO training results are poor (~30% F1 score, frequent generation of random garbage) incidating that the agent is not effectively distinguishing between or executing the recall/engram actions correctly. Training is extremely slow and reosurce-intensive.
- The trained PPO agent code has not yet been integrated with the core graph system.
- No specific test cases for trained PPO agent or its interaction with the graph.
- Multishot prompting for creating the training dataset yielded low variability
- Significant learning regarding PPO implementation, PyTorch, and practical realities of LLM fine-tuning complexity.


## ORIENT

### Strengths
- Project addresses crucial gap in LLMs, core goals remain consistent.
- Decision to separate the models based on initial failed attempts is a pragmatic response to empirical results.
- Foundation is laid for RL, with PPO training infrastructure implemented
- Clear documented learnings from PPO, PyTorch, and finetuning

### Areas for Improvement
- Most significant implementation gap is the lack of integration between newly trained RL policy and dynamic graph system. Project's core hypothesis cannot be tested or validatedu until these elements interact.
- Current PPO training results are major blockers, and policy needs to reliably perform the coreactions with much higher fidelity and efficiency before it can be effectively integrated with the graph
- Current evaluation of PPO agent is disconnected from project's defined success metrics and eval environment. Clearer link is needed to measure meaningful progress
- Issues with synthetic dataset generation directly impact training of RL policy 
- Limited compute is a persistent, severe constraint that impacts progress on training.
- Second loss term in mathematical formulation needs clearer definition on how non-differentiable loss will be incorporated into gradient-based optimization process or its role in its overall system should be reevaluated.
- Report clarity and structure is improved but could use more work, especially in the connection between mathematical formulation, algorithmic components, and implementation progress.

### Critical Risks/Assumptions
- Still unproven whether even a highly accurate PPO agent can effectively manage nad improve LKG structure towrads goal.
- New architecture relies heavily on the router and the effective specialization of the recall/engram models. Coordinating can introduce new complexities.
- Current data generation issues are a risk
- Computational bottlenecks still exist and prevent key development and eval steps.

## DECIDE
- Main priority is addressing the core RL policy issues. Before integration, focusing on making PPO agent reliable and efficient.
- Debug and optimize PPO code to reduce training time.
- Improve PPO agent's accuracy in generating correct recall/engram signals.
- Ensure PPO agent outputs are reliably parsable into the desired format for graph commands.
- Next priority is improving dataset generation and developing a more robust method for creating synthetic training data for RL policy.
- Also need compute resources
- Next, implement model separation and routing, and separate the policy into separate recall and engram models and built routing mechanism.
- After integration, need to reevaluate and measure.

## ACT
### Resource Needs
- GPU compute
- Debugging time
- Dataset generation expertise
- Architecture design time

### Leverage Learnings
- Use PyTorch or Python profiling tools to identify bottlenecks in PPO training loop.
- Experiment with PPO hyperparameters, policy network architectures, and reward signal shaping.
- Prompt engineering to help LLMs reliably generate reliable structured outputs for memory commands.
- If PPO optimization fails, research more efficient RL libraries or algorithms.
- Prepare game eval environment to quickly test the integrated system component.