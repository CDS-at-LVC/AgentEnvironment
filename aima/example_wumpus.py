from aima.agents import Agent
from aima.logic4e import Expr, PropKB, equiv, expr, implies, new_disjunction, pl_resolution
from aima.search import astar_search
from example_environments import Breeze, Bump, Glitter, Scream, Stench, Pit, Wumpus
from example_search import PlanRoute
# _____________________________________________________________________________
# 7.2 The Wumpus World


# Expr functions for WumpusKB and HybridWumpusAgent


def facing_east(time):
    return Expr('FacingEast', time)


def facing_west(time):
    return Expr('FacingWest', time)


def facing_north(time):
    return Expr('FacingNorth', time)


def facing_south(time):
    return Expr('FacingSouth', time)


def wumpus(x, y):
    return Expr('W', x, y)


def pit(x, y):
    return Expr('P', x, y)


def breeze(x, y):
    return Expr('B', x, y)


def stench(x, y):
    return Expr('S', x, y)


def wumpus_alive(time):
    return Expr('WumpusAlive', time)


def have_arrow(time):
    return Expr('HaveArrow', time)


def percept_stench(time):
    return Expr('Stench', time)


def percept_breeze(time):
    return Expr('Breeze', time)


def percept_glitter(time):
    return Expr('Glitter', time)


def percept_bump(time):
    return Expr('Bump', time)


def percept_scream(time):
    return Expr('Scream', time)


def move_forward(time):
    return Expr('Forward', time)


def shoot(time):
    return Expr('Shoot', time)


def turn_left(time):
    return Expr('TurnLeft', time)


def turn_right(time):
    return Expr('TurnRight', time)


def ok_to_move(x, y, time):
    return Expr('OK', x, y, time)


def location(x, y, time=None):
    if time is None:
        return Expr('L', x, y)
    else:
        return Expr('L', x, y, time)


""" [Figure 7.13]
Simple inference in a wumpus world example
"""
wumpus_world_inference = expr("(B11 <=> (P12 | P21))  &  ~B11")

# ______________________________________________________________________________
# 7.7 Agents Based on Propositional Logic
# 7.7.1 The current state of the world


class WumpusKB(PropKB):
    """
    Create a Knowledge Base that contains the atemporal "Wumpus physics" and temporal rules with time zero.
    """

    def __init__(self, dimrow):
        super().__init__()
        self.dimrow = dimrow
        self.tell(~wumpus(1, 1))
        self.tell(~pit(1, 1))

        for y in range(1, dimrow + 1):
            for x in range(1, dimrow + 1):

                pits_in = list()
                wumpus_in = list()

                if x > 1:  # West room exists
                    pits_in.append(pit(x - 1, y))
                    wumpus_in.append(wumpus(x - 1, y))

                if y < dimrow:  # North room exists
                    pits_in.append(pit(x, y + 1))
                    wumpus_in.append(wumpus(x, y + 1))

                if x < dimrow:  # East room exists
                    pits_in.append(pit(x + 1, y))
                    wumpus_in.append(wumpus(x + 1, y))

                if y > 1:  # South room exists
                    pits_in.append(pit(x, y - 1))
                    wumpus_in.append(wumpus(x, y - 1))

                self.tell(equiv(breeze(x, y), new_disjunction(pits_in)))
                self.tell(equiv(stench(x, y), new_disjunction(wumpus_in)))

        # Rule that describes existence of at least one Wumpus
        wumpus_at_least = list()
        for x in range(1, dimrow + 1):
            for y in range(1, dimrow + 1):
                wumpus_at_least.append(wumpus(x, y))

        self.tell(new_disjunction(wumpus_at_least))

        # Rule that describes existence of at most one Wumpus
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                for u in range(1, dimrow + 1):
                    for v in range(1, dimrow + 1):
                        if i != u or j != v:
                            self.tell(~wumpus(i, j) | ~wumpus(u, v))

        # Temporal rules at time zero
        self.tell(location(1, 1, 0))
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                self.tell(implies(location(i, j, 0), equiv(percept_breeze(0), breeze(i, j))))
                self.tell(implies(location(i, j, 0), equiv(percept_stench(0), stench(i, j))))
                if i != 1 or j != 1:
                    self.tell(~location(i, j, 0))

        self.tell(wumpus_alive(0))
        self.tell(have_arrow(0))
        self.tell(facing_east(0))
        self.tell(~facing_north(0))
        self.tell(~facing_south(0))
        self.tell(~facing_west(0))

    def make_action_sentence(self, action, time):
        actions = [move_forward(time), shoot(time), turn_left(time), turn_right(time)]

        for a in actions:
            if action is a:
                self.tell(action)
            else:
                self.tell(~a)

    def make_percept_sentence(self, percept, time):
        # Glitter, Bump, Stench, Breeze, Scream
        flags = [0, 0, 0, 0, 0]

        # Things perceived
        if isinstance(percept, Glitter):
            flags[0] = 1
            self.tell(percept_glitter(time))
        elif isinstance(percept, Bump):
            flags[1] = 1
            self.tell(percept_bump(time))
        elif isinstance(percept, Stench):
            flags[2] = 1
            self.tell(percept_stench(time))
        elif isinstance(percept, Breeze):
            flags[3] = 1
            self.tell(percept_breeze(time))
        elif isinstance(percept, Scream):
            flags[4] = 1
            self.tell(percept_scream(time))

        # Things not perceived
        for i in range(len(flags)):
            if flags[i] == 0:
                if i == 0:
                    self.tell(~percept_glitter(time))
                elif i == 1:
                    self.tell(~percept_bump(time))
                elif i == 2:
                    self.tell(~percept_stench(time))
                elif i == 3:
                    self.tell(~percept_breeze(time))
                elif i == 4:
                    self.tell(~percept_scream(time))

    def add_temporal_sentences(self, time):
        if time == 0:
            return
        t = time - 1

        # current location rules
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                self.tell(implies(location(i, j, time), equiv(percept_breeze(time), breeze(i, j))))
                self.tell(implies(location(i, j, time), equiv(percept_stench(time), stench(i, j))))

                s = list()

                s.append(
                    equiv(
                        location(i, j, time), location(i, j, time) & ~move_forward(time) | percept_bump(time)))

                if i != 1:
                    s.append(location(i - 1, j, t) & facing_east(t) & move_forward(t))

                if i != self.dimrow:
                    s.append(location(i + 1, j, t) & facing_west(t) & move_forward(t))

                if j != 1:
                    s.append(location(i, j - 1, t) & facing_north(t) & move_forward(t))

                if j != self.dimrow:
                    s.append(location(i, j + 1, t) & facing_south(t) & move_forward(t))

                # add sentence about location i,j
                self.tell(new_disjunction(s))

                # add sentence about safety of location i,j
                self.tell(
                    equiv(ok_to_move(i, j, time), ~pit(i, j) & ~wumpus(i, j) & wumpus_alive(time))
                )

        # Rules about current orientation

        a = facing_north(t) & turn_right(t)
        b = facing_south(t) & turn_left(t)
        c = facing_east(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_east(time), a | b | c)
        self.tell(s)

        a = facing_north(t) & turn_left(t)
        b = facing_south(t) & turn_right(t)
        c = facing_west(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_west(time), a | b | c)
        self.tell(s)

        a = facing_east(t) & turn_left(t)
        b = facing_west(t) & turn_right(t)
        c = facing_north(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_north(time), a | b | c)
        self.tell(s)

        a = facing_west(t) & turn_left(t)
        b = facing_east(t) & turn_right(t)
        c = facing_south(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_south(time), a | b | c)
        self.tell(s)

        # Rules about last action
        self.tell(equiv(move_forward(t), ~turn_right(t) & ~turn_left(t)))

        # Rule about the arrow
        self.tell(equiv(have_arrow(time), have_arrow(t) & ~shoot(t)))

        # Rule about Wumpus (dead or alive)
        self.tell(equiv(wumpus_alive(time), wumpus_alive(t) & ~percept_scream(time)))

    def ask_if_true(self, query):
        return pl_resolution(self, query)


# ______________________________________________________________________________


class WumpusPosition:
    def __init__(self, x, y, orientation):
        self.X = x
        self.Y = y
        self.orientation = orientation

    def get_location(self):
        return self.X, self.Y

    def set_location(self, x, y):
        self.X = x
        self.Y = y

    def get_orientation(self):
        return self.orientation

    def set_orientation(self, orientation):
        self.orientation = orientation

    def __eq__(self, other):
        if (other.get_location() == self.get_location() and
                other.get_orientation() == self.get_orientation()):
            return True
        else:
            return False


# ______________________________________________________________________________
# 7.7.2 A hybrid agent


class HybridWumpusAgent(Agent):
    """An agent for the wumpus world that does logical inference. [Figure 7.20]"""

    def __init__(self, dimentions):
        self.dimrow = dimentions
        self.kb = WumpusKB(self.dimrow)
        self.t = 0
        self.plan = list()
        self.current_position = WumpusPosition(1, 1, 'UP')
        super().__init__(self.execute)

    def execute(self, percept):
        self.kb.make_percept_sentence(percept, self.t)
        self.kb.add_temporal_sentences(self.t)

        temp = list()

        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                if self.kb.ask_if_true(location(i, j, self.t)):
                    temp.append(i)
                    temp.append(j)

        if self.kb.ask_if_true(facing_north(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'UP')
        elif self.kb.ask_if_true(facing_south(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'DOWN')
        elif self.kb.ask_if_true(facing_west(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'LEFT')
        elif self.kb.ask_if_true(facing_east(self.t)):
            self.current_position = WumpusPosition(temp[0], temp[1], 'RIGHT')

        safe_points = list()
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                if self.kb.ask_if_true(ok_to_move(i, j, self.t)):
                    safe_points.append([i, j])

        if self.kb.ask_if_true(percept_glitter(self.t)):
            goals = list()
            goals.append([1, 1])
            self.plan.append('Grab')
            actions = self.plan_route(self.current_position, goals, safe_points)
            self.plan.extend(actions)
            self.plan.append('Climb')

        if len(self.plan) == 0:
            unvisited = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    for k in range(self.t):
                        if self.kb.ask_if_true(location(i, j, k)):
                            unvisited.append([i, j])
            unvisited_and_safe = list()
            for u in unvisited:
                for s in safe_points:
                    if u not in unvisited_and_safe and s == u:
                        unvisited_and_safe.append(u)

            temp = self.plan_route(self.current_position, unvisited_and_safe, safe_points)
            self.plan.extend(temp)

        if len(self.plan) == 0 and self.kb.ask_if_true(have_arrow(self.t)):
            possible_wumpus = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if not self.kb.ask_if_true(wumpus(i, j)):
                        possible_wumpus.append([i, j])

            temp = self.plan_shot(self.current_position, possible_wumpus, safe_points)
            self.plan.extend(temp)

        if len(self.plan) == 0:
            not_unsafe = list()
            for i in range(1, self.dimrow + 1):
                for j in range(1, self.dimrow + 1):
                    if not self.kb.ask_if_true(ok_to_move(i, j, self.t)):
                        not_unsafe.append([i, j])
            temp = self.plan_route(self.current_position, not_unsafe, safe_points)
            self.plan.extend(temp)

        if len(self.plan) == 0:
            start = list()
            start.append([1, 1])
            temp = self.plan_route(self.current_position, start, safe_points)
            self.plan.extend(temp)
            self.plan.append('Climb')

        action = self.plan[0]
        self.plan = self.plan[1:]
        self.kb.make_action_sentence(action, self.t)
        self.t += 1

        return action

    def plan_route(self, current, goals, allowed):
        problem = PlanRoute(current, goals, allowed, self.dimrow)
        return astar_search(problem).solution()

    def plan_shot(self, current, goals, allowed):
        shooting_positions = set()

        for loc in goals:
            x = loc[0]
            y = loc[1]
            for i in range(1, self.dimrow + 1):
                if i < x:
                    shooting_positions.add(WumpusPosition(i, y, 'EAST'))
                if i > x:
                    shooting_positions.add(WumpusPosition(i, y, 'WEST'))
                if i < y:
                    shooting_positions.add(WumpusPosition(x, i, 'NORTH'))
                if i > y:
                    shooting_positions.add(WumpusPosition(x, i, 'SOUTH'))

        # Can't have a shooting position from any of the rooms the Wumpus could reside
        orientations = ['EAST', 'WEST', 'NORTH', 'SOUTH']
        for loc in goals:
            for orientation in orientations:
                shooting_positions.remove(WumpusPosition(loc[0], loc[1], orientation))

        actions = list()
        actions.extend(self.plan_route(current, shooting_positions, allowed))
        actions.append('Shoot')
        return actions

# ______________________________________________________________________________
# A simple KB that defines the relevant conditions of the Wumpus World as in Fig 7.4.
# See Sec. 7.4.3
wumpus_kb = PropKB()

P11, P12, P21, P22, P31, B11, B21 = expr('P11, P12, P21, P22, P31, B11, B21')
wumpus_kb.tell(~P11)
wumpus_kb.tell(B11 | '<=>' | (P12 | P21))
wumpus_kb.tell(B21 | '<=>' | (P11 | P22 | P31))
wumpus_kb.tell(~B11)
wumpus_kb.tell(B21)
