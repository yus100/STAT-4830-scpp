# Self-Critique: MAGS - Dynamic Memory Modules for Large Language Models via Liquid Knowledge Graphs

## OBSERVE

### Initial Reactions
- The report presents a clear and detailed technical approach but is dense and difficult to parse for new readers.
- The mathematical formulation is useful but could be better connected to the implementation specifics.
- Code implementation is functional but faces issues with JSON formatting and instruction-following in model responses.
- Inference stability was observed, but resource utilization could be optimized further.
- RL training is incomplete, limiting full evaluation.
- While there are strong ambitious goals for the MAGS system, currently, the implementation is a very basic graph RAG, lacking a lot of core dynamic and learning mechanisms.
- Some core functionality has been implemented: foundational graph RAG that allows node query, addition, and removal; custom game environment and dataloaders for testing
- The current implementation is slow, despite low CPU utilization, suggesting bottlenecks in the LLM inference process -- there are also no formal performance metrics yet.
- Currently, the "engram" stage is nondeterministic and adds random or wrong information that doesn't directly pertain to the prompt.
- The LLM doesn't always adhere to the required output formats, even at low temperatures, which impacts our ability to automate memory modifications.
- The basic testing environment is set up without formal metrics yet.
- Limited compute is a bottleneck currently.


## ORIENT

### Strengths
- The proposal clearly defines limitations in existing LLMs and offers a structured, neuroscience-based approach.
- The proposal covers essential aspects such as RL, graph structures, and long-term memory consolidation.
- Initial implementations show promising memory retrieval and knowledge persistence capabilities.
- The proposed mechanisms (dual memory blocks, recall/engram steps, dynamic graph algorithms, RL guidance, contradiction handling) are conceptually sound and draw from relevant research areas.
- The report articulates the problem clearly and proposes concrete metrics for measuring success, centered around a novel evaluation method.
- Despite lacking core MAGS logic, the development of a basic graph interface and testing environment demonstrates concrete progress towards establishing the necessary infrastructure.
- The focus on inspectable and explainable memory is a significant strength for research.

### Areas for Improvement
- Crucially, there is a significant gap between the technical plan and the current basic implementation. The core novelty of MAGS lies in LKG dynamics and RL training, which aren't yet functional.
- The engram step is unreliable currently; the current failure of the LLM to accurately and reliably generate memory updates is a major blocker. We need to focus efforts on prompt engineering, instruction-tuning, or refining RL signals specific to engram accuracy.
- The current slow inference speed makes iterative development and testing difficult -- we need to identify bottlenecks and optimize them.
- While there are defined metrics, we are not yet applying them to measure progress or guide iteration, and testing currently is manual.
- The report definitely needs structural refinement and clearer explanations of how different components interact. We can include visual aids and link mathematical formulations more explicitly to the proposed algorithms.
- We need to ensure our synthetic data is good quality, diverse, and scale sufficient for training and evaluation.

### Critical Risks/Assumptions
- The central hypothesis is currently unproven. We have not been able to test our hypothesis at this stage, so the potential failure case is significant.
- Training GRPO and LoRA to reliably optimize both recall and engram steps is a complex task with no convergence guarantees.
- Our algorithms like our "Game of Life" for edge weighting, k-core decomposition, and pruning must remain computationally efficient as the graph grows.
- The entire engram update mechanism depends on the LLM reliably producing output parseable into graph commands, so the current inconsistency is a major risk until addressed.
- Current compute resources (half CPU cores, limited GPU availability) may not be sufficient for full-scale training.

## DECIDE
- We want to prioritize the core MAGS implementation from here, shifting from basic RAG functionality to implementing the unique components of MAGS outlined in the technical approach, specifically:
    - Anchor node selection and DFS traversal for recall
    - Hebbian ede strengthening and recency correction
    - Basic pruning based on edge weights
    - Initial implementatino of KEDGKG for contradiction handling
- We want to profile the current setup to understand why inference is slow and address the bottlenecks
- We want to refine the engram generation and parsing by improving our prompt engineering or investigating methods to ensure the LLM reliably produces the JSON format for memory updates.
- Looking to find something to secure compute resources.
- Start building tools and processes for generating synthetic datasets for training and eval.
- Begin integrating the proposed metrics into the testing framework to move beyond manual review and gain objective measures of progress on specific components as implemented
- Lower priority -- refine the report  structure and clarity 

## ACT
### Resource Needs
- GPU compute
- Development time
- Graph processing library proficiency
- Dataset generation tools

### Leverage Learnings
- The experience with local models forms a basis, but challenges with non-determinism and structured output highlight then ee for careful prompt design and maybe stronger base models or specific finetuning.
- Recognizing the computational bottleneck early 
- Understanding the gap between theoretical design and piece-by-piece implementation