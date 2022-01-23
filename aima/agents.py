from aima.thing import Thing
from statistics import mean

import random
import copy
import collections

class Agent(Thing):
    """An Agent is a subclass of Thing with one required instance attribute 
    (aka slot), .program, which should hold a function that takes one argument,
    the percept, and returns an action. (What counts as a percept or action 
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method. If it were a method, then the
    program could 'cheat' and look at aspects of the agent. It's not supposed
    to do that: the program can only look at the percepts. An agent program
    that needs a model of the world (and of the agent itself) will have to
    build and maintain its own model. There is an optional slot, .performance,
    which is a number giving the performance measure of the agent in its
    environment."""

    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0
        if program is None or not isinstance(program, collections.abc.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

        self.program = program

    def can_grab(self, thing):
        """Return True if this agent can grab this thing.
        Override for appropriate subclasses of Agent and Thing."""
        return False


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


# ______________________________________________________________________________


def TableDrivenAgentProgram(table):
    """
    [Figure 2.7]
    This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it, provide as table a dictionary of all
    {percept_sequence:action} pairs.
    """
    percepts = []

    def program(percept):
        percepts.append(percept)
        action = table.get(tuple(percepts))
        return action

    return program


def RandomAgentProgram(actions):
    """An agent that chooses an action at random, ignoring all percepts.
    >>> list = ['Right', 'Left', 'Suck', 'NoOp']
    >>> program = RandomAgentProgram(list)
    >>> agent = Agent(program)
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1, 0): 'Clean' , (0, 0): 'Clean'}
    True
    """
    return lambda percept: random.choice(actions)


# ______________________________________________________________________________


def SimpleReflexAgentProgram(rules, interpret_input):
    """
    [Figure 2.10]
    This agent takes action based solely on the percept.
    """

    def program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        action = rule.action
        return action

    return program


def ModelBasedReflexAgentProgram(rules, update_state, model):
    """
    [Figure 2.12]
    This agent takes action based on the percept and state.
    """

    def program(percept):
        program.state = update_state(program.state, program.action, percept, model)
        rule = rule_match(program.state, rules)
        action = rule.action
        return action

    program.state = program.action = None
    return program


def rule_match(state, rules):
    """Find the first rule that matches state."""
    for rule in rules:
        if rule.matches(state):
            return rule


# ______________________________________________________________________________


# ______________________________________________________________________________


def compare_agents(EnvFactory, AgentFactories, n=10, steps=1000):
    """See how well each of several agents do in n instances of an environment.
    Pass in a factory (constructor) for environments, and several for agents.
    Create n instances of the environment, and run each agent in copies of
    each one for steps. Return a list of (agent, average-score) tuples.
    >>> environment = TrivialVacuumEnvironment
    >>> agents = [ModelBasedVacuumAgent, ReflexVacuumAgent]
    >>> result = compare_agents(environment, agents)
    >>> performance_ModelBasedVacuumAgent = result[0][1]
    >>> performance_ReflexVacuumAgent = result[1][1]
    >>> performance_ReflexVacuumAgent <= performance_ModelBasedVacuumAgent
    True
    """
    envs = [EnvFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(envs)))
            for A in AgentFactories]


def test_agent(AgentFactory, steps, envs):
    """Return the mean score of running an agent in each of the envs, for steps
    >>> def constant_prog(percept):
    ...     return percept
    ...
    >>> agent = Agent(constant_prog)
    >>> result = agent.program(5)
    >>> result == 5
    True
    """

    def score(env):
        agent = AgentFactory()
        env.add_thing(agent)
        env.run(steps)
        return agent.performance

    return mean(map(score, envs))

class Direction:
    """A direction class for agents that want to move in a 2D plane
        Usage:
            d = Direction("down")
            To change directions:
            d = d + "right" or d = d + Direction.R #Both do the same thing
            Note that the argument to __add__ must be a string and not a Direction object.
            Also, it (the argument) can only be right or left."""

    R = "right"
    L = "left"
    U = "up"
    D = "down"

    def __init__(self, direction):
        self.direction = direction

    def __add__(self, heading):
        """
        >>> d = Direction('right')
        >>> l1 = d.__add__(Direction.L)
        >>> l2 = d.__add__(Direction.R)
        >>> l1.direction
        'up'
        >>> l2.direction
        'down'
        >>> d = Direction('down')
        >>> l1 = d.__add__('right')
        >>> l2 = d.__add__('left')
        >>> l1.direction == Direction.L
        True
        >>> l2.direction == Direction.R
        True
        """
        if self.direction == self.R:
            return {
                self.R: Direction(self.D),
                self.L: Direction(self.U),
            }.get(heading, None)
        elif self.direction == self.L:
            return {
                self.R: Direction(self.U),
                self.L: Direction(self.D),
            }.get(heading, None)
        elif self.direction == self.U:
            return {
                self.R: Direction(self.R),
                self.L: Direction(self.L),
            }.get(heading, None)
        elif self.direction == self.D:
            return {
                self.R: Direction(self.L),
                self.L: Direction(self.R),
            }.get(heading, None)

    def move_forward(self, from_location):
        """
        >>> d = Direction('up')
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (0, -1)
        >>> d = Direction(Direction.R)
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (1, 0)
        """
        # get the iterable class to return
        iclass = from_location.__class__
        x, y = from_location
        if self.direction == self.R:
            return iclass((x + 1, y))
        elif self.direction == self.L:
            return iclass((x - 1, y))
        elif self.direction == self.U:
            return iclass((x, y - 1))
        elif self.direction == self.D:
            return iclass((x, y + 1))
