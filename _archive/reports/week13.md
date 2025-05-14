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

No change.

### Algorithm and Implementation Choice

No Change.


### Validation Methods and Resource Requirements

Used some textworld games to help validate, namely difficult and hard versions of a text-maze and a kitchen-task that seemed easy and hard for some other state of the art algorithms.
---

## Initial Results

### Working Implementation

Wrapped up most of the remaining code. Made a new little demo website where you can play around with the hyperparamaters and mess around with different world loads and the actual graph itself. This mainly uses the OpenAI backend but can also be set to our endpoint since it uses the same schema.

Also did some hyperparamater turning on the final model. It wasn't very great since our loss function isn't differientable so we cant use gradient descent, so I just did a grid search. However, each iteration took a few hours to finish, so I basically just chose a few values for each of the major paramaters, chose the one that minimized loss holding the other steady, and then moved to the next one (so probably not the real minimum loss across the grid, since I basically did it by hand).

Also made some visualizations to see the reconciliation and the pruning / scaling in action!

### Performance Metrics

The main performance metric here was from the hyperparamater turning, although the less wasn't much better than our random initial values (about 9-10% better)/


### Test Cases

Created a test case demo to set up the Liquid Knowledge Graph and make sure everything is imported and working correctly. Comes with a nice UI :)

### Limitations

We didn't have time to implement a lot of the stuff that we wanted to. This includes stuff like advanced decomposition / consolidation (whichn my not be easy or possible with triplet schema)

We also never were able to try a bigger model, or retrain on higher fidelity datasets.

We also never replaced chat history with a 'user' noun for infinite scaling (in theory)

Most of the things in our intitial abstract's reach goals we never reach unfortunately.

### Resource Usage
Hyperparamater turning was very slow (running on CPU and unoptimizined Metal CPU shaders), but not very memory intensive. 

### Unexpected Challenges

Honestly, main challenges was just getting everything done in time. We had a lot more bugs than we anticipated, and we were just smoothing everything up before the demo, since we have a lot of exams and can't really work the next few days. Other than that, no major challenges.

## Next Steps

### Immediate Improvements

- Last week, so no more improvements that we can do! I think we built a pretty good prototype, and we'll probably work on it in the future.

### Technical Challenges to Address

- Main things is optizimg it for CUDA / Nvidia GPUs if we can to make ot train and inference a lot faster, esepcially for our fine-tuned models.

### Questions for Help
 - N/A
### Alternative Approaches to Try
- Last week so also N/a
### What we've learned so far
- Learned a whole lot about how LLMs work, esepcially the weird quirks in training/fine-tuning and inference across architectures (and how slow it can be!). Made me appreciate all the engineering that goes into them a lot more. Also learned that it helps a lot to make your loss function differentiable lol. 