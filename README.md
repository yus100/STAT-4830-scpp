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

Repository Structure and main components
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
- Run the GRPO Pipeline in `src/RL/GRPO.ipynb` for the single model. Alternatively, for the dual models, run the pipelines in `notebooks/Ask_Model_Final_Training.ipynb` and `notebooks/Write_Model_Final_Training.ipynb` (Note these pieplines cannot be run on Colab, they will crash due to memory unless you use a paid model. To use xformers you must use a windows laptop or intel mac locally). Once the model(s) finishes, download the GGUF to `v2/mags/inference/models/` with the name `gemma-3-finetuned.gguf` (for single model) or `gemma-3-ask-final` and `gemma-3-write-final` (for dual models). The current setup should finish in about 50 mins (although the accuracy for these hyperparamaters are not great. For full performance use `r=64` `alpha=128` and `epochs=10` and full precision model). 

Alternatively, install the files from here [single-model link](https://drive.google.com/drive/folders/1tEI30F7rcOWQ2hn-mwwjgBpS9zN4KdsW?usp=sharing) or [dual-model ask](https://huggingface.co/Pulkith/gemma-3-ask-float16) and [dual-model write](https://huggingface.co/Pulkith/gemma-3-ask-float16) and follow the same instructions. You can also install just the LoRA adapators [dual-model-adaptors](https://huggingface.co/Pulkith/gemma-3-ask-finetune) and make the model locally if you have installation errors using the script in `notebooks/LORA to VLLM.ipynb` Sorry! It was to big to push to github.
- Run `src/mags_test_inference.py` to make sure your setup succeeded (all test cases should pass like below)

<img width="1843" alt="Screenshot 2025-05-14 at 12 42 14" src="https://github.com/user-attachments/assets/d3515758-f529-46c7-bde2-b7e52047c36c" />

- Install VLLM with your correct drivers here [link](https://docs.vllm.ai/en/stable/getting_started/installation.html)
- Run the interactive terminal version at `src\v2\gen_pipeline.py`. If your VLLM or model is not configured, it falls back to Generic (Cloud) LLM (Not Fine-tuned instead prompted, in which case you need to pass your API key)

Sorry for not using Colab. It was very frustrating, with the sessions constantly terminating (and me losing all my checkpoints or training runs), or running out of memory (and even disk storage?), running out of compute/cooldown for multiple daysand crashing randomly. It also wasn't must faster than an intel mac with metal GPU optimization. I also worked all semester doing regular python library-type code, so it wasn't easy to port it over to a notebook, and I felt like the current terminal experience is pretty good too!!

Also please check out the code, I think it's pretty modular and scalable :)

## Demo
- Insall and run the 'Go Live' feature in VSCode to run the demo on `src/demo_site/index3.html`. This uses the lightweight prompted version for the router so you'll need your API key. The other site uses the API for the full model, but it is a pain to set up (email me pulkith [at] wharton.upenn.edu if you want to however!). Please let me know if you have any issues with any of this as well!
