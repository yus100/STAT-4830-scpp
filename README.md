## MAGS: Memory Augmented Graph Scaling
Enabling LLMs to learn

> Major Note: Since we pivoted our idea during Week 4 / 5, we have ~2 weeks less progress, and do not have the code/slide/docs from Week 3 / 4 in our archive. However, we tried our best to still have our finished prototype by the end of the semester!

## Team
- Pulkith Paruchuri
- Stanley Yu
- Pragya Singh
- Shailesh Kumar

## Summary
> A full report can be found at [docs/report.pdf](docs/report.pdf) and [docs/Final_Presentation.pdf](docs/Final_Presentation.pdf).

Current LLMs struggle with learning and long-term retention, relying on inefficient and ephemeral context windows that limit adaptability and scalability. MAGS (Memory Augmented Generative System) proposes a neuroscience-inspired architecture using Liquid Knowledge Graphs (LKGs) and dual memory blocks—episodic (eMB) and semantic (sMB)—to emulate human memory and enable continuous, test-time learning. The system dynamically updates memory through a reinforcement-learning-trained model and Hebbian plasticity-inspired methods, managing graph growth to ensure efficiency. We tested the architecture on a combination of text-based and logic games to measure accuracy, adaptability, and explainability. We found promising results that match or exceed state-of-the-art approaches in many tasks.

## Structure
AI Model Structure (Conceptual)
```
├── _archive/                    # Previous work and documentation
├── docs/                        # Previous work and documentation
│   ├── report.pdf               # Final Paper
│   └── Final_Presentation.pdf   # Final slides
├── src/                         # Python source code and modules
│   ├── RL                       # GRPO and PPO Pipelines
│   ├── v2                       # Main Test Pipelines
│   └── demo_site                # Main interactive site
└── notebooks/                   # Final demo notebooks (inferences the `src/` modules)
```
## Setup
- Create a virtual environment with python 3.12 (required)
- Install dependencies: `pip install -r requirements.txt`
- All code was run with a CPU and Metal GPU. It should work with any GPU and should be automatically detected
- Install LLama 3.2-3B Q6 from hugging face
- Run `src/RL/PPO.py` to train the model with the directory to this model (Sorry, too large to push to Github). This will take about 30 hours on CPU, or about 12-15 hours on a Metal GPU (likely same on Colab T4 ). You can reduce the hyperparamaters to speed up training.
- Run `src/pipeline.py` to evaluate the model (or use the notebooks)

## Demo
- Run `npm install` to install dependencies
- Run `npm run dev` or use the Go Live extension / feature in VSCode to run the demo
