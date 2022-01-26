# AgentEnvironment
An example based on the Agent/Environment python code from AIMA. I've restructured the book code and added a matplotlib-based visualiazation so that we can work with it outside of Jupyter.

The example code (room.py, simple_ex.py) solves all of our first homework coding exercise except for the "stop moving when the floor is clean" requirement. That requirement is intended to provoke you -- it requires that the vacuum has some understanding of the entire environment outside of its percepts. You can solve it by adding, e.g., a parameter to the agent that tells it the initial number of dirty squares. Not realistic.

Remember, the steps to getting this running are
* clone the repo
* open with vscode, open a terminal, use **python -m venv venv** to create a virtual environment. 
* close that terminal and open a new one; vscode should activate venv for you. 
* update pip with **python -m pip install --upgrade pip**
* install requirements **pip install -r requirements.txt** (you can trim requirements.txt first if you want. Lots of stuff there you don't need)
* Now the program should run.

# Examples
The examples that are currently present are:
* vacuum_search.py, which creates a Problem representing a linear vacuum environment so that we can apply various uninformed search algorithms towards finding an optimal action sequence.
