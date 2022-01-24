# AgentEnvironment
An example based on the Agent/Environment python code from AIMA. I've restructured the book code and added a matplotlib-based visualiazation so that we can work with it outside of Jupyter.

The example code (room.py, simple_ex.py) solves all of our first homework coding exercise except for the "stop moving when the floor is clean" requirement. That requirement is intended to provoke you -- it requires that the vacuum has some understanding of the entire environment outside of its percepts. You can solve it by adding, e.g., a parameter to the agent that tells it the initial number of dirty squares. Not realistic.
