import numbers
import random
from aima.agents import Agent, Direction
from aima.thing import Thing
from aima.utils import distance_squared, turn_heading

from ipythonblocks import BlockGrid
from IPython.display import HTML, display, clear_output
from time import sleep

import numpy as np
import matplotlib.pyplot as plt



class Environment:
    """Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []  # List of classes that can go into environment

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def is_done(self):
        """By default, we're done when we can't find a live agent."""
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive:
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)
            self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for _ in range(steps):
            if self.is_done():
                return
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        if isinstance(location, numbers.Number):
            return [thing for thing in self.things
                    if thing.location == location and isinstance(thing, tclass)]
        return [thing for thing in self.things
                if all(x == y for x, y in zip(thing.location, location)) and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)"""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            thing.location = location if location is not None else self.default_location(thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class DisplayEnvironment(Environment):
    """An environment that augments the basic step with a couple of
    display_*() calls, which derived classes can override to draw 
    the environment in some way.
    
    The object should initialize the display in the display_init()
    method. This method is called before the first step is done in run().
    it should display the initial state of the environment.

    display_step() should be overriden to erase the previous state and 
    draw the new state. 

    The derived class GridEnvironmentMPL shows how to do this using 
    matplotlib to draw a 2D grid environment (ala our vacuum cleaner) 
    environment. 
    """

    def __init__(self, width, height, delay=1):
        super().__init__()
        self.width = width
        self.height = height
        self.display_delay = delay

    def display_init(self):
        """Override this to display the initial environment. This is
        called at the beginning of run(). """
        raise NotImplementedError

    def display_step(self):
        """Override this to display a new state for the environment. 
        This is called at the end of step()"""
        raise NotImplementedError

    def step(self):
        super().step()
        self.display_step()
        sleep(self.display_delay)

    def run(self, steps=1000):
        self.display_init()
        super().run(steps)


class GridEnvironmentMPL(DisplayEnvironment):
    """
    An 2D grid environment that uses matplotlib to draw itself at each
    step.
    The environment has Things and Agents in it, of course. Override the 
    agent_color() and thing_color() methods to draw the elements in 
    specific colors.  The default is that all things are green and the agent
    is red.
    """

    def thing_color(self, thing):
        return 'green'

    def agent_color(self, agent):
        return 'red'

    def window_close(self, event):
        self.figure.close()
        exit()

    def display_init(self):
        self.step_num = 0
        # put matplotlib in interactive mode
        plt.ion()
        
        # grab the "figure" from plt so we can manipulate it later
        self.figure, _ = plt.subplots()

        # set up the basic plot
        plt.rcParams["figure.figsize"] = [10.0, 10.0]
        plt.rcParams["figure.autolayout"] = True

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xticks(range(self.width+1))
        plt.yticks(range(self.height+1))
        plt.grid()

        #grab the locations of every thing in the environment. 
        x, y = zip(*[[x-0.5 for x in t.location] for t in self.things if not isinstance(t, Agent)])
        thing_colors = [self.thing_color(t) for t in self.things if not isinstance(t, Agent)]
        self.thing_scatter = plt.scatter(x, y, marker="o", s=100, c=thing_colors)

        #same for agents
        x, y = zip(*[[x-0.5 for x in a.location] for a in self.agents])
        agent_colors = [self.agent_color(a) for a in self.agents]
        self.agent_scatter = plt.scatter(x, y, marker="o", s=100, c=agent_colors)

        # label the figure
        self.figure.suptitle('Initial State')

        #draw our figure in a matplotlib window
        self.figure.canvas.mpl_connect('close_event', self.window_close)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def display_step(self):
        self.step_num += 1

        #grab the locations of every thing in the environment. 
        x, y = zip(*[[x-0.5 for x in t.location] for t in self.things if not isinstance(t, Agent)])
        thing_colors = [self.thing_color(t) for t in self.things if not isinstance(t, Agent)]

        self.thing_scatter.set_offsets(np.c_[x, y])
        self.thing_scatter.set_facecolor(thing_colors)

        #same for agents
        x, y = zip(*[[x-0.5 for x in a.location] for a in self.agents])
        agent_colors = [self.agent_color(a) for a in self.agents]
        self.agent_scatter.set_offsets(np.c_[x, y])
        self.agent_scatter.set_facecolor(agent_colors)

        # label the figure
        self.figure.suptitle(f'Step {self.step_num}')

        #draw our figure in a matplotlib window
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


class XYEnvironment(Environment):
    """This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.

    Agents perceive things within a radius. Each agent in the
    environment has a .location slot which should be a location such
    as (0, 1), and a .holding slot, which should be a list of things
    that are held."""

    def __init__(self, width=10, height=10):
        super().__init__()

        self.width = width
        self.height = height
        self.observers = []
        # Sets iteration start and end (no walls).
        self.x_start, self.y_start = (0, 0)
        self.x_end, self.y_end = (self.width, self.height)

    perceptible_distance = 1

    def things_near(self, location, radius=None):
        """Return all things within radius of location."""
        if radius is None:
            radius = self.perceptible_distance
        radius2 = radius * radius
        return [(thing, radius2 - distance_squared(location, thing.location))
                for thing in self.things if distance_squared(
                location, thing.location) <= radius2]

    def percept(self, agent):
        """By default, agent perceives things within a default radius."""
        return self.things_near(agent.location)

    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'TurnRight':
            agent.direction += Direction.R
        elif action == 'TurnLeft':
            agent.direction += Direction.L
        elif action == 'Forward':
            agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
        elif action == 'Grab':
            things = [thing for thing in self.list_things_at(agent.location) if agent.can_grab(thing)]
            if things:    
                agent.holding.append(things[0])
                print("Grabbing ", things[0].__class__.__name__)
                self.delete_thing(things[0])
        elif action == 'Release':
            if agent.holding:
                dropped = agent.holding.pop()
                print("Dropping ", dropped.__class__.__name__)
                self.add_thing(dropped, location=agent.location)

    def default_location(self, thing):
        location = self.random_location_inbounds()
        while self.some_things_at(location, Obstacle):
            # we will find a random location with no obstacles
            location = self.random_location_inbounds()
        return location

    def move_to(self, thing, destination):
        """Move a thing to a new location. Returns True on success or False if there is an Obstacle.
        If thing is holding anything, they move with him."""
        thing.bump = self.some_things_at(destination, Obstacle)
        if not thing.bump:
            thing.location = destination
            for o in self.observers:
                o.thing_moved(thing)
            for t in thing.holding:
                self.delete_thing(t)
                self.add_thing(t, destination)
                t.location = destination
        return thing.bump

    def add_thing(self, thing, location=None, exclude_duplicate_class_items=False):
        """Add things to the world. If (exclude_duplicate_class_items) then the item won't be
        added if the location has at least one item of the same class."""
        if location is None:
            super().add_thing(thing)
        elif self.is_inbounds(location):
            if (exclude_duplicate_class_items and
                    any(isinstance(t, thing.__class__) for t in self.list_things_at(location))):
                return
            super().add_thing(thing, location)

    def is_inbounds(self, location):
        """Checks to make sure that the location is inbounds (within walls if we have walls)"""
        x, y = location
        return not (x < self.x_start or x > self.x_end or y < self.y_start or y > self.y_end)

    def random_location_inbounds(self, exclude=None):
        """Returns a random location that is inbounds (within walls if we have walls)"""
        location = (random.randint(self.x_start, self.x_end),
                    random.randint(self.y_start, self.y_end))
        if exclude is not None:
            while location == exclude:
                location = (random.randint(self.x_start, self.x_end),
                            random.randint(self.y_start, self.y_end))
        return location

    def delete_thing(self, thing):
        """Deletes thing, and everything it is holding (if thing is an agent)"""
        if isinstance(thing, Agent):
            del thing.holding

        super().delete_thing(thing)
        for obs in self.observers:
            obs.thing_deleted(thing)

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))
        for y in range(1, self.height - 1):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

        # Updates iteration start and end (with walls).
        self.x_start, self.y_start = (1, 1)
        self.x_end, self.y_end = (self.width - 1, self.height - 1)

    def add_observer(self, observer):
        """Adds an observer to the list of observers.
        An observer is typically an EnvGUI.

        Each observer is notified of changes in move_to and add_thing,
        by calling the observer's methods thing_moved(thing)
        and thing_added(thing, loc)."""
        self.observers.append(observer)

    def turn_heading(self, heading, inc):
        """Return the heading to the left (inc=+1) or right (inc=-1) of heading."""
        return turn_heading(heading, inc)


class Obstacle(Thing):
    """Something that can cause a bump, preventing an agent from
    moving into the same square it's in."""
    pass


class Wall(Obstacle):
    pass


# ______________________________________________________________________________


class GraphicEnvironment(XYEnvironment):
    def __init__(self, width=10, height=10, boundary=True, color={}, display=False):
        """Define all the usual XYEnvironment characteristics,
        but initialise a BlockGrid for GUI too."""
        super().__init__(width, height)
        self.grid = BlockGrid(width, height, fill=(200, 200, 200))
        if display:
            self.grid.show()
            self.visible = True
        else:
            self.visible = False
        self.bounded = boundary
        self.colors = color

    def get_world(self):
        """Returns all the items in the world in a format
        understandable by the ipythonblocks BlockGrid."""
        result = []
        x_start, y_start = (0, 0)
        x_end, y_end = self.width, self.height
        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.list_things_at((x, y)))
            result.append(row)
        return result

    """
    def run(self, steps=1000, delay=1):
        "" "Run the Environment for given number of time steps,
        but update the GUI too." ""
        for step in range(steps):
            sleep(delay)
            if self.visible:
                self.reveal()
            if self.is_done():
                if self.visible:
                    self.reveal()
                return
            self.step()
        if self.visible:
            self.reveal()
    """

    def run(self, steps=1000, delay=1):
        """Run the Environment for given number of time steps,
        but update the GUI too."""
        for _ in range(steps):
            self.update(delay)
            if self.is_done():
                break
            self.step()
        self.update(delay)

    def update(self, delay=1):
        sleep(delay)
        self.reveal()

    def reveal(self):
        """Display the BlockGrid for this world - the last thing to be added
        at a location defines the location color."""
        self.draw_world()
        # wait for the world to update and
        # apply changes to the same grid instead
        # of making a new one.
        clear_output(1)
        self.grid.show()
        self.visible = True

    def draw_world(self):
        self.grid[:] = (200, 200, 200)
        world = self.get_world()
        for x in range(0, len(world)):
            for y in range(0, len(world[x])):
                if len(world[x][y]):
                    self.grid[y, x] = self.colors[world[x][y][-1].__class__.__name__]

    def conceal(self):
        """Hide the BlockGrid for this world"""
        self.visible = False
        display(HTML(''))


# ______________________________________________________________________________
# Continuous environment

class ContinuousWorld(Environment):
    """Model for Continuous World"""

    def __init__(self, width=10, height=10):
        super().__init__()
        self.width = width
        self.height = height

    def add_obstacle(self, coordinates):
        self.things.append(PolygonObstacle(coordinates))


class PolygonObstacle(Obstacle):

    def __init__(self, coordinates):
        """Coordinates is a list of tuples."""
        super().__init__()
        self.coordinates = coordinates
