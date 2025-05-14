# STAT 4830 Project Repository

Welcome to your project repository! This template helps you develop and implement an optimization project over the semester.

## Spring 2025 Project Teams

Current student projects:

1. **GraphSAGE-Enhanced Cold-Start Recommendation System**  
   * **Summary:**  
     This report presents a graph‐based recommendation system designed for e-commerce cold‐start scenarios. It integrates heterogeneous product metadata—from images to reviews—by leveraging GraphSAGE for modeling inter‐item relationships and CLIP embeddings for visual clustering. The document details the problem formulation, experimental evaluations, and resource considerations for delivering personalized recommendations to users with minimal history.  
   * **Link:** [GraphSAGE-Enhanced Cold-Start Recommendation System](https://github.com/kuomat/STAT-4830-vllm-project/blob/main/report.md)

2. **Efficient Transformer Attention via Custom Sparse Mask Learning**  
   * **Summary:**  
     Addressing the quadratic complexity of standard Transformer attention, this report introduces a method to learn custom sparse attention masks. By minimizing the KL-divergence between a modified (sparse) attention model and a baseline Transformer, the study demonstrates how selective token attention can preserve performance while reducing computational overhead. The report outlines the mathematical formulation, implementation in PyTorch, and experimental results on language modeling tasks.  
   * **Link:** [Efficient Transformer Attention via Custom Sparse Mask Learning](https://github.com/charisgao/STAT-4830-project/blob/main/report.md)

3. **Reinforcement Learning for Poker Strategy Optimization**  
   * **Summary:**  
     This report explores the use of reinforcement learning to optimize poker strategies in a simulated environment. It details the design of the simulation, reward structures tailored for bluffing and incomplete information, and experiments comparing learned policies against baseline strategies. The study highlights challenges such as non-deterministic decision-making and proposes refinements to enhance the AI's competitive performance.  
   * **Link:** [Reinforcement Learning for Poker Strategy Optimization](https://github.com/AC2005/STAT-4830-poker/blob/main/Report.md)

4. **Portfolio Optimization with Multi-Objective Constraints**  
   * **Summary:**  
     This report outlines a quantitative framework for daily portfolio construction that simultaneously maximizes risk-adjusted returns and minimizes drawdown while accounting for transaction costs. It formulates the optimization problem mathematically—incorporating the Sharpe ratio, maximum drawdown, and weight-change penalties—and implements a gradient-based solution using PyTorch. The report details validation via backtesting on historical data, discusses challenges such as handling short positions and computational constraints, and concludes with next steps for scaling the strategy with more assets and advanced risk measures.  
   * **Link:** [Portfolio Optimization with Multi-Objective Constraints](https://github.com/dhruv575/STAT-4830-project-base/blob/main/report.md)

5. **Collaborative Optimization in Group Decision-Making: Week 3 Progress Report**  
   * **Summary:**  
     This Week 3 progress report details a collaborative approach to group decision-making optimization. It outlines initial model proposals, experimental setups, and preliminary results that integrate individual preferences into a unified recommendation system. The document discusses the challenges of merging diverse opinions and sets the stage for further refinements in collaborative optimization techniques.  
   * **Link:** [Collaborative Optimization in Group Decision-Making: Week 3 Progress Report](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/Week%203%20Report.pdf)

6. **Parameter-Efficient Reinforcement Learning via Curriculum Strategies**  
   * **Summary:**  
     This report presents a curriculum learning framework aimed at achieving parameter efficiency in reinforcement learning. It outlines strategies for incrementally increasing model complexity while keeping computational costs low and demonstrates through benchmark experiments how such an approach can improve convergence and overall performance with fewer parameters.  
   * **Link:** [Parameter-Efficient Reinforcement Learning via Curriculum Strategies](https://github.com/JustinSQiu/STAT-4830-curriculum-learning-project/blob/main/docs/Parameter_Efficient_Reinforcement_Learning_Paper.pdf)

7. **Optimizing Concession Stand Placement for Enhanced Stadium Accessibility and Revenue**  
   * **Summary:**  
     This report formulates a spatial optimization model for the strategic placement of concession stands in stadiums. By combining heuristic search with mathematical optimization, it seeks to balance user accessibility with revenue generation. Simulation studies and performance metrics validate the model's effectiveness, offering insights into its practical deployment in large venues.  
   * **Link:** [Optimizing Concession Stand Placement for Enhanced Stadium Accessibility and Revenue](https://github.com/awu626/STAT-4830-project/blob/main/report.md)

8. **Multi-Agent Reinforcement Learning for Optimized Decision-Making**  
   * **Summary:**  
     Focusing on multi-agent systems, this report explores reinforcement learning techniques to enhance decision-making in competitive and cooperative environments. It details a framework where agents learn optimal strategies through interaction, addresses challenges such as non-stationarity and credit assignment, and presents simulation results that highlight improvements in collective performance.  
   * **Link:** [Multi-Agent Reinforcement Learning for Optimized Decision-Making](https://github.com/sheyanlalmohammed1/STAT-4830-CTP-RL-project/blob/main/report.md)

9. **Optimizing Sleep Stage Classification through Integrated Physiological Signal Analysis**  
   * **Summary:**  
     This report presents a machine learning framework designed to classify sleep stages by integrating diverse physiological signals. It details advanced data preprocessing, feature extraction techniques, and classification algorithms aimed at accurately differentiating between wakefulness and various sleep states, underscoring the clinical potential of enhanced sleep analysis methodologies.  
   * **Link:** [Optimizing Sleep Stage Classification through Integrated Physiological Signal Analysis](https://github.com/kimberlyliang/STAT-4830-GOALZ-project/blob/main/report.md)

10. **Optimizing Urban Travel for Electric Vehicles in NYC**  
    * **Summary:**  
      This report develops an optimization framework targeting improved travel efficiency for electric vehicles in NYC. It integrates real-time traffic data, charging station locations, and route optimization algorithms to demonstrate significant reductions in travel time and energy consumption, paving the way for smarter urban mobility solutions.  
    * **Link:** [Optimizing Urban Travel for Electric Vehicles in NYC](https://github.com/TheCrypted/STAT-4830-project-base/blob/main/report.md)

11. **Real-Time Predictive Modeling for 1v1 Basketball Live-Streams**  
    * **Summary:**  
      This report details the development of a real-time predictive modeling system for 1v1 basketball live-streams. It combines statistical analysis with machine learning techniques to forecast game outcomes on the fly, addressing challenges such as latency and rapidly changing game conditions, and highlighting innovative approaches to delivering timely and accurate sports analytics.  
    * **Link:** [Real-Time Predictive Modeling for 1v1 Basketball Live-Streams](https://github.com/fortyjmps/your-repo-stat4830/blob/main/report.md)

## Getting Started

1. **Finding Your Project Idea**
   - Start with our [Project Ideas Guide](docs/finding_project_ideas.md)
   - Use AI to explore and refine your ideas
   - Take time to find something you care about

   It's very important you learn to use AI tools in your work! [Noam Brown](https://x.com/polynoamial/status/1870307185961386366) (OpenAI) says that students should...
   > Practice working with AI. Human+AI will be superior to human or AI alone for the foreseeable future. Those who can work most effectively with AI will be the most highly valued.

   ![Noam tweet](figures/noam.png)

2. **Week 3 Deliverable**
   - Follow the [Week 3 Instructions](docs/assignments/week3_deliverable_instructions.md)
   - Required components:
     - Initial report draft
     - Self-critique document analyzing your report's strengths and weaknesses
     - Supporting Jupyter notebooks/code
   - Due: Friday, January 31, 2025

## Project Development Cycle

Each week follows an OODA (Observe, Orient, Decide, Act) loop that helps you improve your project systematically:

![Project Development Cycle - A diagram showing the OODA loop (Observe, Orient, Decide, Act) adapted for project development. Each phase has specific activities: Observe (Review Report, Check Results), Orient (Write Critique, Find Gaps), Decide (Plan Changes, Set Goals), and Act (Code, Run Tests). The phases are connected by arrows showing the flow of work, with a feedback loop labeled "Iterative Development" completing the cycle.](docs/figures/ooda_loop.png)

Each cycle produces specific deliverables:
- OBSERVE: Updated report draft
- ORIENT: Self-critique document
- DECIDE: Next actions plan
- ACT: Code changes & results

See the [Week 3 Instructions](docs/assignments/week3_deliverable_instructions.md) for detailed guidance on writing your first self-critique.

## Project Schedule

### Deliverables (Due Fridays)
- Week 2 (Jan 24): Email Project Team Names to yihuihe@wharton.upenn.edu
- Week 3 (Jan 31): Report Draft 1 + Code + Self Critique
- Week 4 (Feb 7): Slides Draft 1
- Week 5 (Feb 14): Report Draft 2 + Code + Self Critique
- Week 6 (Feb 21): Slides Draft 2
- Week 7 (Feb 28): Report Draft 3 + Code + Self Critique
- Week 8: ⚡ Lightning Talks in Class (Mar 5/7) & Slides Draft 3 due Friday ⚡
- Spring Break (Mar 8-16)
- Week 9 (Mar 21): Report Draft 4 + Code + Self Critique
- Week 10 (Mar 28): Slides Draft 4
- Week 11 (Apr 4): Report Draft 5 + Code + Self Critique
- Week 12 (Apr 11): Slides Draft 5
- Week 13: Final Presentations in Class (Apr 24/29) & Report Draft 6 + Code + Self Critique due Friday (Apr 18)
- Week 14 (Apr 29): Final Report + Code + Self Critique Due

Note: Instructions for peer feedback will be added throughout the semester for each deliverable.

Each draft builds on the previous one, incorporating feedback and new results. You'll meet with course staff three times during the semester to discuss your progress.

## Project Grading

Each deliverable is graded on five components:
- Report (20%): Problem statement, methodology, results
- Implementation (35%): Working code, tests, experiments
- Development Process (15%): Logs, decisions, iterations
- Critiques (15%): Reflection and planning
  - Self-critiques (required)
  - Peer critiques (when assigned)
  - Response to feedback
- Repository Structure (15%): Organization, documentation, clarity

Remember:
- Quality > Quantity
- Working > Perfect

## Repository Structure

```
your-repo/
├── README.md                    # This file
├── report.md                    # Your project report
├── notebooks/                   # Jupyter notebooks
├── src/                        # Source code
├── tests/                      # Test files
└── docs/
    ├── finding_project_ideas.md    # Guide to finding your project
    ├── assignments/                # Assignment instructions
    ├── llm_exploration/           # AI conversation logs
    └── development_log.md         # Progress & decisions
```

## Development Environment

### Editor Setup
We recommend using one of these editors:

1. **VS Code** (Free, Industry Standard)
   - Download from https://code.visualstudio.com/
   - Install recommended extensions:
     - Python
     - GitHub Pull Requests
     - GitHub Copilot (FREE for students!)
       - Sign up at https://education.github.com/discount_requests/application
       - This gives you FREE access to GitHub Copilot
       - Plus other GitHub student benefits

2. **Cursor** (Paid Alternative, $20/month)
   - Built on VS Code with additional AI features
   - Same interface and shortcuts as VS Code
   - Same extensions work
   - Added AI assistance for code exploration

Both editors work well with Git and provide excellent AI assistance. VS Code with Copilot is recommended for beginners as it's free with your student status and is the industry standard.

### Required Tools
- Python 3.10+
- PyTorch
- Jupyter Notebook/Lab
- Git

## Git Setup and Workflow

### First Time Setup
1. Fork this repository
   - Click "Fork" in the top right
   - Name it `STAT-4830-[team-name]-project`
   - This creates your own copy that can receive updates

2. Set up Git (if you haven't already):
   If you installed VS Code or Cursor, they'll help you install Git! Both editors have excellent Git integration built in.
   
   For detailed instructions, see the [Official Git installation guide](https://github.com/git-guides/install-git)

   After installing, set up your identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@upenn.edu"
   ```

3. Clone your fork:
   ```bash
   # HTTPS (easier):
   git clone https://github.com/[your-username]/STAT-4830-[team-name]-project.git

   # SSH (if you've set up SSH keys):
   git clone git@github.com:[your-username]/STAT-4830-[team-name]-project.git
   
   cd STAT-4830-[team-name]-project
   ```

4. Add upstream remote (to get updates):
   ```bash
   # HTTPS:
   git remote add upstream https://github.com/damek/STAT-4830-project-base.git

   # SSH:
   git remote add upstream git@github.com:damek/STAT-4830-project-base.git
   ```

5. Add your team members as collaborators:
   - Go to your repo on GitHub
   - Settings → Collaborators → Add people
   - Add using their GitHub usernames

### Working on Your Project
1. Create a new branch:
   ```bash
   git checkout -b exploration
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin exploration
   ```

### Getting Updates
When the base repository is improved:
```bash
# Get updates
git fetch upstream
git checkout main
git merge upstream/main

# Update your branch
git checkout exploration
git merge main
```

### Troubleshooting
- Having Git issues? Post on Ed Discussion
- Can't push/pull? Check if you're using HTTPS or SSH
- Windows path too long? Enable long paths:
  ```bash
  git config --system core.longpaths true
  ```

## Getting Help
- Use AI tools (ChatGPT, GitHub Copilot)
- See course staff for technical issues
- Document your progress






