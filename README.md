# AgentEnvironment
An example based on the Agent/Environment python code from AIMA. I've restructured the book code and added a matplotlib-based visualiazation so that we can work with it outside of Jupyter.

So, this repo is growing to become home for a variety of rewritten/reorganized bits of code from the book's (_Artificial Intelligence: A Modern Approach_, 4<sup>th</sup> edition) python repo. 

Remember, the steps to getting this running are
* clone the repo
* open with vscode, open a terminal, use **python -m venv venv** to create a virtual environment. 
* close that terminal and open a new one; vscode should activate venv for you. 
* update pip with **python -m pip install --upgrade pip**
* install requirements **pip install -r requirements.txt** (you can trim requirements.txt first if you want. Lots of stuff there you don't need)
* Now the program should run.

# Changes to book's repo

My approach has been to move python source over from the book's repo as we need it. I'm placing it in the [aima](aima) module. 

The authors of the aima python codebase combined a lot of stuff together in single python files; for instance, the original **agents.py** contained the agent class, the environment class, several variant subclasses of those, and several examples. 

I've decided to split these things up. Fundamental base classes (e.g., **Agent**, **Problem**) have gotten their own files, examples have been split off into separate files. I think this is making for a collection of files that is a bit easier to navigate, though as things progress we might want to reorganize into submodules.

The base folder in the repo contains examples that I created for our class conversation, or that are related to our homework.

# Our Examples
The examples that are currently present are:
* [simple_ex](simple_ex.py), a simple agent in a simple environment with a simple program to run.
* [room](room.py), an environment that shows off the matplotlib-based visualization of gridlike environments that I added.
* [vacuum search](vacuum_search.py), which creates a [Problem](aima/problem.py) representing a linear vacuum environment so that we can apply various uninformed search algorithms towards finding an optimal action sequence.
* [eight_queens](eight_queens.py), an example application of hill-climbing and simulated annealing to the classic 8-queens problem.
