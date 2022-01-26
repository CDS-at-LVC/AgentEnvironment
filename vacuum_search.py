from aima.environments import Environment
from aima.agents import Agent
from aima.search import Problem, SimpleProblemSolvingAgentProgram
from aima.search import breadth_first_graph_search, breadth_first_tree_search
from aima.search import depth_first_graph_search, depth_first_tree_search
from aima.thing import Thing


class VacuumProblem(Problem):
    """
    We record a state as a list of 4 characters. The first three are 'D' 
    for dirty and 'C'  for clean, indicating whether the left, middle, or
    right gridpoints are dirty or clean. The fourth entry is 0, 1, 2 
    indicating the position of the vacuum. 
    """
    def __init__(self, initial, goal, size):
        super().__init__(initial, goal)
        self.size = size

    def actions(self, state):
        """
        The list of possible actions is always the same
        """
        return ['R', 'L', 'S']

    def result(self, state, action):
        """
        compute the state that results from one of 'R', 'L', or 'S'
        """
        new_state = list(state)
        if action == 'L' and state[-1] > 0:
            new_state[-1] -= 1            
        elif action == 'R' and state[-1] < self.size-1:
            new_state[-1] += 1
        elif action == 'S':
            new_state[state[-1]] = 'C'
        return tuple(new_state)


class VacuumProblemSolverProgram(SimpleProblemSolvingAgentProgram):
    def __init__(self, size, initial):
        super().__init__(initial)
        self.size = size

    def update_state(self, state, percept):
        return self.state

    def formulate_goal(self, state):
        return [tuple('C' for _ in range(self.size)) + tuple([x]) for x in range(self.size)]

    def formulate_problem(self, state, goal):
        return VacuumProblem(initial=state, goal=goal, size=self.size)

    def search(self, problem):
        return depth_first_graph_search(problem=problem).solution()


class Dirt(Thing):
    pass


class ThreeSquareEnv(Environment):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def percept(self, agent):
        return self.list_things_at(agent.location)
    
    def execute_action(self, agent, action):
        if action == 'S':
            items = self.list_things_at(agent.location, tclass=Dirt)
            if items:
                print(f'{str(agent)} cleaned {str(items[0])} at location: {agent.location}')
                # remove the dirt
                self.delete_thing(items[0]) 
        elif action == 'L' and agent.location > 0:
            print('moving left')
            agent.location -= 1
        elif action == 'R' and agent.location < self.size - 1:
            print('moving right')
            agent.location += 1
        else:
            agent.alive = False
        

if __name__=='__main__':
    env = ThreeSquareEnv(size=7)
    env.add_thing(Dirt(), 0)
    env.add_thing(Dirt(), 2)
    env.add_thing(Dirt(), 3)
    env.add_thing(Dirt(), 5)
    env.add_thing(Dirt(), 6)
    
    vacuum = Agent(VacuumProblemSolverProgram(size=7, initial=('D','C','D','D','C','D','D', 3)))
    env.add_thing(vacuum, 3)

    env.run(50)

    