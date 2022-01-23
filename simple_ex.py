from aima.agents import Agent
from room import Dirt, Obstacle, Room
import random


class Vacuum(Agent):
    """Our vacuum agent needs nothing except its program, since 
    it's a simple reflex agent."""

    # a list of visited cells. We know these are clean
    visited = []

    def __init__(self, w, h):
        self.width = w
        self.height = h

        super().__init__(lambda percepts: self.myprogram(percepts))

    def myprogram(self, percepts):
        # update our model - a list of cells we have visited. Our 
        # location is part of what we perceive, so this is essentially
        # a list of percepts.
        if self.location not in self.visited:
            self.visited.append(self.location)

        for thing in percepts:
            if isinstance(thing, Dirt):
                return "suck"

        unvisited_dir = [dir for dir,cell in self.get_neighbors().items() 
                            if cell not in self.visited]

        if unvisited_dir:        
            return random.choice(unvisited_dir)
        else:
            return random.choice(['R', 'L', 'U', 'D'])

    def get_neighbors(self):
        neighbors = {}
        if self.location[0] > 1:
            neighbors['U'] = (self.location[0], self.location[1]+1)
        if self.location[0] < self.height:
            neighbors['D'] = (self.location[0], self.location[1]-1)
        if self.location[1] > 1:
            neighbors['L'] = (self.location[0]-1, self.location[1])
        if self.location[1] < self.width:
            neighbors['R'] = (self.location[0]+1, self.location[1])
        return neighbors

# The main program
if __name__ == '__main__':
    room = Room(10, 10, 3)
    vacuum = Vacuum(10, 10)
    room.add_thing(Dirt(), (4, 2))
    room.add_thing(Dirt(), (2, 5))
    room.add_thing(Dirt(), (0, 7))
    room.add_thing(Dirt(), (5, 6))
    room.add_thing(Dirt(), (5, 4))
    room.add_thing(Dirt(), (6, 5))
    room.add_thing(Obstacle(), (4, 1))
    room.add_thing(Obstacle(), (7, 5))
    room.add_thing(Obstacle(), (2, 2))
    room.add_thing(Obstacle(), (5, 7))
    room.add_thing(Obstacle(), (7, 1))

    room.add_thing(vacuum, (5, 5))

    room.run(200)