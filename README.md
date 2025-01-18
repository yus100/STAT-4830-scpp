# STAT-4830 Project Base

> This is in-progress scaffolding for the Week 3 project deliverable. Like the rest of the course, I'm developing this in the open. The structure will evolve based on student needs.

## Base Repository Overview
This repository helps you:
- Organize your project exploration
- Document your llm conversations (seriously, use an llm a lot)
- Have a working repo from day one

## Repository Structure
```
project_template/
├── docs/                  # Your documentation
│   └── llm_exploration/   # AI interaction logs
├── notebooks/             # Jupyter notebooks for exploration
└── README.md             # This guide
```

---

# Your Team Project Starts Here
After forking this repository, replace everything below this line with your team's content.

## Team Members
- Name 1 (GitHub username)
- Name 2 (GitHub username)
- Name 3 (GitHub username)

> Note: Feel free to use an anonymous GitHub account, just make sure to share your GitHub username and Penn identity with the course instructor and TA.

## Project Overview
[2-3 sentences about your project idea. Don't stress too much about this - your direction will probably change as you learn more about optimization]

## Initial Direction
- What problem are you trying to solve?
- Why is this an optimization problem?
- What's your initial approach?
- What don't you know yet? (probably a lot, that's fine)

## Current Status
[What's working, what isn't, what you're stuck on - be honest!]

---

# Reference: Development Setup

## Development Environment
I use [Cursor](https://cursor.sh/) for this course. It's basically VS Code with additional AI features built in:
- Same interface and shortcuts as VS Code
- Same extensions work
- Added AI assistance for code exploration and debugging
- If you learn Cursor, you're also learning VS Code

Cursor is a paid product ($20/month). For a free alternative with similar capabilities:

[VS Code](https://code.visualstudio.com/) is the industry standard that Cursor builds on:
- Download from https://code.visualstudio.com/
- It comes with Git integration
- Install these extensions:
  - Python
  - GitHub Pull Requests
  - GitHub Copilot (FREE for students!)
    - Sign up at https://education.github.com/discount_requests/application
    - This gives you FREE access to GitHub Copilot
    - Plus other GitHub student benefits

Both editors will work great for the course. VS Code with Copilot gives you professional-grade AI assistance at no cost (thanks to your student status), while Cursor offers a different take on AI integration. Use what works for you!

## Git Setup and Workflow

### First Time Setup
1. Fork this repository
   - Click "Fork" in the top right
   - Name it `STAT-4830-[team-name]-project`
   - This creates your own copy that can receive updates

2. Set up Git (if you haven't already):
   
   If you installed VS Code or Cursor, they'll help you set up Git! The process is almost identical in both editors.
   
   For detailed instructions, see:
   - [Official Git installation guide](https://github.com/git-guides/install-git)
   - The guide includes VS Code-specific instructions
   
   After installing, just set up your identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@upenn.edu"
   ```

3. Clone your fork:
   - Open Terminal (Mac/Linux) or Git Bash (Windows)
   - Clone the repo:
     ```bash
     # HTTPS (easier):
     git clone https://github.com/[your-username]/STAT-4830-[team-name]-project.git

     # SSH (if you've set up SSH keys):
     git clone git@github.com:[your-username]/STAT-4830-[team-name]-project.git
     ```
   - Move into the directory:
     ```bash
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

### Start Working
1. Create a new branch for your work:
   ```bash
   git checkout -b exploration
   ```

2. Start exploring:
   - Log your ChatGPT conversations in `docs/llm_exploration/`
   - Get ANY optimization example running in the notebook
   - Document what you learn

3. Save your changes:
   ```bash
   git add .
   git commit -m "Add initial exploration"
   git push origin exploration
   ```

### Getting Updates
When we improve the base repository:
```bash
# Get updates
git fetch upstream
git checkout main
git merge upstream/main

# Update your exploration branch
git checkout exploration
git merge main
```

### Troubleshooting
- Having git issues? Post on Ed Discussion
- Can't push/pull? Check if you're using HTTPS or SSH
- Windows path too long? Enable long paths:
  ```bash
  git config --system core.longpaths true
  ``` 