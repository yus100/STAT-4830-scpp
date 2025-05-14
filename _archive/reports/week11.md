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

We switched to using a triplet-based system, instead of having just nodes store large texts, and edges that were meaninglesss. Triplets were from a paper we found recently, and we realized it would be a lot more effective to work with, and would help with a lot of graph management instead of using some less accurate methods or a LLM to solve them. Basically triplets store information in the form of (verb, action, object) like (John, likes, apples) as (Node, edge, Node). It makes the graph a lot mor dense, but not that much more larger (since we store much less text in each node). However, it likely will increase the computational complexity. 

We also switched to using Gemma 4B! It's a bit better, and seems to be better at instruction following and for our needs.


### Validation Methods and Resource Requirements

No change (still local plus a little bit of Colab for unsloth).
---

## Initial Results

### Working Implementation

We finished most of the initial graph algorithms, like decay, and strengthing, but have more to work on in terms of things like consolidation and reconicliation (right now we just add more instead of addressing contradicting infromtion) which is in `/src`. 

We also trained the new Gemma models namely Gemma 1B and Gemma 4B in `/trained_models/`. 

We also have a working implmementation with triplets, and rebased everything to work with our new schema.

We created some new larger datasets as well for this new schema, which I think helped when we trained the Gemma models in a slight performance boost.

Broke down our model to be seperate for recall and engram, seems to be performing a lot better, and actually outputting the correct information instead of mixing them up.

### Performance Metrics

Hard to quantify the results from the graph, but with the RAG we seme to have maybe a 5% improvement. However, this is likely because we're stuffing a lot more information in the content window rather than intelligently do it.

We also have edges and nodes get pruned and become permenant over time, so its cool to see it happen. We don't really have the right paramaters as of yet, so not really sure, but will probably do some grid search to find more optimal paramaters and to get some real metrics.


### Test Cases

As mentioned previously, no test cases.

### Limitations

We don't have the right hyperparamaters for our graph, so accuracy is still suffering until we find the optimal configuration.

We realized the model has a small context window, so it can't do large paragraphs of text. We probably won't address this before the semester ends, but just something we realized.

### Resource Usage

The actual algorithm itself (since its trained) doesn't really take any resource usages (pretty marginal) in the other of MB for RAM for the graph, and some spikes in CPU usage during inference time for the model, but nothing when its idle.

### Unexpected Challenges

We don't know how to deal with triplets with the same verb right now (like John | has | a car, and John | has | a truck). Is this conflicting or different information? How
do we reconcicle this. Not sure as of yet.

Develoepd the graph, but forgot to figure out how to actually validate it. Running our evaluation set takes many hours, so its hard to find a good configuration from a search.
Trying to think of a better way but unsure as of right now.

## Next Steps

### Immediate Improvements

- Finishing the graph algorithms to include everything we originally mentioned
- Making using of the semantic vs episodic memory somehow? Unsure of how right now.
- Doing a grid search on the graph to find the right paramaters
- Figure out how to consolidate and reconcile information.

### Technical Challenges to Address

- How do we reconcile information. We're trying to think of a fast/effecient approach, but many need to just rely on an LLM
- How do we effeciently consolidate information? Before with the text-nodes it was simply using an LLM, but with triplets, it likely isn't possible because
the schema is so rigit. We likely just need to rely on pruning for memory usage.

### Questions for Help
 - Mainly just the two questions above. Any insight would be super help in answering these!
### Alternative Approaches to Try
- May want to use textworld games. We sya a paper where it mentioned how its a good way to test long-term memory because of how complex they are!
### What we've learned so far
- Learned a lot from research and implmeneting how Graph RAGs work, and how they traverse to get information. Also this was our first week getting major new results from iterating with using new models and new datasets, and a new schema, so it was also just really cool to see our changes work in real-time after our first approach didn't work that well!