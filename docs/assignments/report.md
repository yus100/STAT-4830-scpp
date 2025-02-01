# Problem Statement (1/2 page)
### What are you optimizing? (Be specific)
We are looking to optimize the process of artificial protein synthesis by using transformers to ideally predict function and form, and generate sequences to meet specific critereon for proteins. 

### Why does this problem matter?
Currently, recent developments in protein syntehsis, such as AlphaFold, ESMFold, and others, have accelerated the process of drug development and protein synthesis significantly. The next step in protein synthesis and similar fields is being able to generate protein sequences to match a specific function, structure, and form, in order to accelerate drug development, protein development, and unlock a host of solutions to biological and medical issues whose solutions have eluded us for years.

### How will you measure success?
Currently, this is one of the biggest challenges we anticipate to see with the project. We can measure success by comparing our genearted protein sequences with existing NCBI and human genome databases, to see whether they match patterns that we can currently find in the human genome. Additionally, we can test by running AlphaFold queries to see whether the generated protein sequences would properly fold, based on AlphaFold's models to predict whether the protein's attributes align.

### What are your constraints?
Our constraints are pretty biologically-based: We want to be able to develop protein sequences that fold correctly, and that ideally match the specified function.

### What data do you need?
A lot of the data we would use for this project come from NCBI databases and human genome banks, which are mostly publicly available. The data is pretty large in size, which may lead to further issues or narrowing in the scope of the project later on. Additionally, if we can find any research or data regarding artificial protein synthesis validation methods, that would also prove helpful.

### What could go wrong?
One of our biggest concerns with carrying out this project is that validation processes are a little sketchy at the moment. While we can compare against NCBI databases and AlphaFold, we have limited ways of checking whether our generated protein sequence folds correctly or not.

# Technical Approach (1/2 page)
### Mathematical formulation (objective function, constraints)
### Algorithm/approach choice and justification
### PyTorch implementation strategy
### Validation methods
### Resource requirements and constraints

# Initial Results (1/2 page)
### Evidence your implementation works
### Basic performance metrics
### Test case results
### Current limitations
### Resource usage measurements
### Unexpected challenges

### Next Steps (1/2 page)
### Immediate improvements needed
### Technical challenges to address
### Questions you need help with
### Alternative approaches to try
### What you've learned so far