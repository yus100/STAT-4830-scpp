# MAGS: Dynamic Memory Modules for Large Language Models via Liquid Knowledge Graphs




## Problem Statement

The problem remains developing an architecture that enables long-term model learning.

### What Are We Optimizing?

No Change.

### Why Does This Problem Matter?

No Change.

### How Will We Measure Success?

Prof. Davis brought up Auren last time, so a bit of our success is now to see if we can replicate it in terms of how well it can get to learn someone. Its hard to quantify this of course but we think it would be a fun challenge after trying out the app! We were able to also hack the app a little so I think we have a solid understanding of how it works, and hopefully our prototype should be a little more accurate (although much slower probably).

### Constraints / Data Requirements

No Change.

### Potential Failures
No Change.

---

## Technical Approach

### Mathematical Formulation

No change.

### Algorithm and Implementation Choice

We mentioned last week that we wanted to switch to two models for the fine-tuned engram/recall. As of now we are still using one model with an exapanded dataset, and on larger paramater models, to see if it yeilds an improveent. Based on our results, we will still be moving to two models, from next week on.

We also switched to GRPO on unsloth, to see how well it worked. It trains much faster, and uses MUCH less memory, and is a lot simpler to use since its not our own code lol. The accuracy improves a little bit, although this is likely because we just expanded our dataset a little bit more as well.

### Validation Methods and Resource Requirements

Still running locally since Colab terminates our session before we can finish fine-tuning. Trying to see if there's any free cloud compute, but we also don't know CUDA and since the learning curve seems to be high we likely will stick with local training.
---

## Initial Results

### Working Implementation
Honestly, didn't have much time to add much more since Spring Break because of midterms and break itself, but we tried a few more things. We expanded our dataset which can be found in `src/RL/data` to have almost double the number of examples, with a little more variety. Also our `src/RL/trained_models` folder has all the trained models in GGUF format with our new models, which were Llama 3B and Qwen and a 0.5B model for fun. 

We finished integrating the model into the graph system right before the weekend and were able to test this against baselines which were just the model and the model + a static RAG. 


### Performance Metrics

The larger models had much better results! We were able to get it up to ~60% adhernece, and about a ~30% improvement over the base model in our senseless game. Using GRPO in unsloth was also much faster, taking only a few mins for a low run, and a few hours for the full run, as opposed to over a day and a half as before!

### Test Cases

As mentioned previously, no test cases.

### Limitations

As for the model, we don't really have any limitations from what we can tell. We want to try the dual model, but we don't think we can improve the accuracy too much

For the actual graph, we need to start implementing all the features we suggested initally, since that is the main part of MAGS.

### Resource Usage

Using unsloth, we use a lot less resources! Our max memory is ~15GB when running on colab, and it takes between 5 mins for the minimal run, and about 3-7 hours for the full run(3 hours for just the first dataset, 7 hours for both). 

### Unexpected Challenges

We didn't realize than unsloth doesn't run on mac, so we had to use colab to train it. However, colab kept constantly disocnnecting randomly, and we kept running out of memory.

It was also alsmost impossible to download the model after training. During the download process it would crash when compiling it, or just not download whne we zip it and press the download button. We had to try multuiple times (and retrain it since the checkpoints disepeared with the when the session crashes), before it worked even once. Was just a large waste of time unfortunately.

## Next Steps

### Immediate Improvements

- As mentiond previously, we want to permenantly switch to the two model approach using the larger models.
- Also want to work on some of the graph part of it, like decay, hebbian plasticity, and traversals on the anchor nodes

### Technical Challenges to Address

- As of right now, we think we are done with the model part, so our main technical hurdles are implementing the graph, and integrating all the new features
- We also want to see if we can have some sort of model checkpointing/switching to compare multiple iterations quickly, but we're unsure of how to do this since loading a new model takes a while.
### Questions for Help
 - As of now, no questions again! We're at a little break in terms of what we finished and what we're about to start, so we probably will have a lot of questions soon though.
### Alternative Approaches to Try
- As mentioned previously, our main approaches likely will just be using the dual-model approach for better success.
### What we've learned so far
- We learned a lot about using GRPO, and how much more quick unsloth (and memory effecient) as opposed to PPO! It was really cool to see it work and offer the same or better results than our own implementations, which honestly just go to show how much more we have to learn!