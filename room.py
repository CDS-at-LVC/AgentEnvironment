from black import re
from aima.thing import Thing
from aima.environments import GridEnvironmentMPL

class Dirt(Thing):
    pass

class Obstacle(Thing):
    pass

class Vacuumed(Thing):
    pass

class Room(GridEnvironmentMPL):
    move_delta = {'U': (0, 1), 'D': (0,-1), 'L': (-1,0), 'R':(1,0)}

    def thing_color(self, thing):
        if isinstance(thing, Dirt):
            return 'brown'
        elif isinstance(thing, Obstacle):
            return 'red'
        elif isinstance(thing, Vacuumed):
            return 'yellow'
        else:
            return 'black'

    def agent_color(self, agent):
        return 'blue'

    def apply_move(self, loc, dir):
        """return a tuple that is in direction dir from loc"""
        return tuple(x + y for x, y in zip(loc, self.move_delta[dir]))
        
    def legal_location(self, loc):
        """checks that loc is within our room""" 
        return 0 < loc[0] <= self.height and 0 < loc[1] <= self.width

    def percept(self, agent):
        """return a list of things that are in our agent's location"""
        things = self.list_things_at(agent.location)
        return things
    
    def move(self, agent, dir):
        """Move the agent in the given direction, if possible.
        return true if the move succeeded."""
        new_loc = self.apply_move(agent.location, dir)
        
        in_bounds = self.legal_location(new_loc)

        hit_obstacle = True
        if (in_bounds and not self.list_things_at(new_loc, Obstacle)):
            agent.location = new_loc
            hit_obstacle = False
        
        return in_bounds and not hit_obstacle


    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action in ['R', 'L', 'U', 'D']:
            print('{} decided to move {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            # move the agent. Might fail
            loc = agent.location
            if self.move(agent, action):
                self.add_thing(Vacuumed(), loc)
                print('\t\tmove succeeded')
            else:
                print('\t\tmove failed')
        elif action == "suck":
            items = self.list_things_at(agent.location, tclass=Dirt)
            if len(items) != 0:
                print('{} cleaned {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                # remove the dirt
                self.delete_thing(items[0]) 

