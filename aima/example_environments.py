import random
from aima.thing import Thing
from aima.environments import Environment, XYEnvironment, Wall
from aima.agents import Agent
from example_agents import Explorer, ReflexVacuumAgent, RandomVacuumAgent
from example_agents import TableDrivenVacuumAgent, ModelBasedVacuumAgent

class Dirt(Thing):
    pass

loc_A, loc_B = (0, 0), (1, 0)  # The two locations for the Vacuum world

class VacuumEnvironment(XYEnvironment):
    """The environment of [Ex. 2.12]. Agent perceives dirty or clean,
    and bump (into obstacle) or not; 2D discrete world of unknown size;
    performance measure is 100 for each dirt cleaned, and -1 for
    each turn taken."""

    def __init__(self, width=10, height=10):
        super().__init__(width, height)
        self.add_walls()

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """The percept is a tuple of ('Dirty' or 'Clean', 'Bump' or 'None').
        Unlike the TrivialVacuumEnvironment, location is NOT perceived."""
        status = ('Dirty' if self.some_things_at(
            agent.location, Dirt) else 'Clean')
        bump = ('Bump' if agent.bump else 'None')
        return status, bump

    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'Suck':
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
        else:
            super().execute_action(agent, action)

        if action != 'NoOp':
            agent.performance -= 1


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])


# ______________________________________________________________________________
# The Wumpus World


class Gold(Thing):
    def __eq__(self, rhs):
        """All Gold are equal"""
        return rhs.__class__ == Gold


class Bump(Thing):
    pass


class Glitter(Thing):
    pass


class Pit(Thing):
    pass


class Breeze(Thing):
    pass


class Arrow(Thing):
    pass


class Scream(Thing):
    pass


class Wumpus(Agent):
    screamed = False


class Stench(Thing):
    pass

class WumpusEnvironment(XYEnvironment):
    pit_probability = 0.2  # Probability to spawn a pit in a location. (From Chapter 7.2)

    # Room should be 4x4 grid of rooms. The extra 2 for walls

    def __init__(self, agent_program, width=6, height=6):
        super().__init__(width, height)
        self.init_world(agent_program)

    def init_world(self, program):
        """Spawn items in the world based on probabilities from the book"""

        "WALLS"
        self.add_walls()

        "PITS"
        for x in range(self.x_start, self.x_end):
            for y in range(self.y_start, self.y_end):
                if random.random() < self.pit_probability:
                    self.add_thing(Pit(), (x, y), True)
                    self.add_thing(Breeze(), (x - 1, y), True)
                    self.add_thing(Breeze(), (x, y - 1), True)
                    self.add_thing(Breeze(), (x + 1, y), True)
                    self.add_thing(Breeze(), (x, y + 1), True)

        "WUMPUS"
        w_x, w_y = self.random_location_inbounds(exclude=(1, 1))
        self.add_thing(Wumpus(lambda x: ""), (w_x, w_y), True)
        self.add_thing(Stench(), (w_x - 1, w_y), True)
        self.add_thing(Stench(), (w_x + 1, w_y), True)
        self.add_thing(Stench(), (w_x, w_y - 1), True)
        self.add_thing(Stench(), (w_x, w_y + 1), True)

        "GOLD"
        self.add_thing(Gold(), self.random_location_inbounds(exclude=(1, 1)), True)

        "AGENT"
        self.add_thing(Explorer(program), (1, 1), True)

    def get_world(self, show_walls=True):
        """Return the items in the world"""
        result = []
        x_start, y_start = (0, 0) if show_walls else (1, 1)

        if show_walls:
            x_end, y_end = self.width, self.height
        else:
            x_end, y_end = self.width - 1, self.height - 1

        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.list_things_at((x, y)))
            result.append(row)
        return result

    def percepts_from(self, agent, location, tclass=Thing):
        """Return percepts from a given location,
        and replaces some items with percepts from chapter 7."""
        thing_percepts = {
            Gold: Glitter(),
            Wall: Bump(),
            Wumpus: Stench(),
            Pit: Breeze()}

        """Agents don't need to get their percepts"""
        thing_percepts[agent.__class__] = None

        """Gold only glitters in its cell"""
        if location != agent.location:
            thing_percepts[Gold] = None

        result = [thing_percepts.get(thing.__class__, thing) for thing in self.things
                  if thing.location == location and isinstance(thing, tclass)]
        return result if len(result) else [None]

    def percept(self, agent):
        """Return things in adjacent (not diagonal) cells of the agent.
        Result format: [Left, Right, Up, Down, Center / Current location]"""
        x, y = agent.location
        result = []
        result.append(self.percepts_from(agent, (x - 1, y)))
        result.append(self.percepts_from(agent, (x + 1, y)))
        result.append(self.percepts_from(agent, (x, y - 1)))
        result.append(self.percepts_from(agent, (x, y + 1)))
        result.append(self.percepts_from(agent, (x, y)))

        """The wumpus gives out a loud scream once it's killed."""
        wumpus = [thing for thing in self.things if isinstance(thing, Wumpus)]
        if len(wumpus) and not wumpus[0].alive and not wumpus[0].screamed:
            result[-1].append(Scream())
            wumpus[0].screamed = True

        return result

    def execute_action(self, agent, action):
        """Modify the state of the environment based on the agent's actions.
        Performance score taken directly out of the book."""

        if isinstance(agent, Explorer) and self.in_danger(agent):
            return
            
        agent.bump = False
        if action in ['TurnRight', 'TurnLeft', 'Forward', 'Grab']:
            super().execute_action(agent, action)
            agent.performance -= 1
        elif action == 'Climb':
            if agent.location == (1, 1):  # Agent can only climb out of (1,1)
                agent.performance += 1000 if Gold() in agent.holding else 0
                self.delete_thing(agent)
        elif action == 'Shoot':
            """The arrow travels straight down the path the agent is facing"""
            if agent.has_arrow:
                arrow_travel = agent.direction.move_forward(agent.location)
                while self.is_inbounds(arrow_travel):
                    wumpus = [thing for thing in self.list_things_at(arrow_travel)
                              if isinstance(thing, Wumpus)]
                    if len(wumpus):
                        wumpus[0].alive = False
                        break
                    arrow_travel = agent.direction.move_forward(agent.location)
                agent.has_arrow = False

    def in_danger(self, agent):
        """Check if Explorer is in danger (Pit or Wumpus), if he is, kill him"""
        for thing in self.list_things_at(agent.location):
            if isinstance(thing, Pit) or (isinstance(thing, Wumpus) and thing.alive):
                agent.alive = False
                agent.performance -= 1000
                agent.killed_by = thing.__class__.__name__
                return True
        return False

    def is_done(self):
        """The game is over when the Explorer is killed
        or if he climbs out of the cave only at (1,1)."""
        explorer = [agent for agent in self.agents if isinstance(agent, Explorer)]
        if len(explorer):
            if explorer[0].alive:
                return False
            else:
                print("Death by {} [-1000].".format(explorer[0].killed_by))
        else:
            print("Explorer climbed out {}."
                  .format("with Gold [+1000]!" if Gold() not in self.things else "without Gold [+0]"))
        return True

