from aima.graph import Graph, GraphProblem, UndirectedGraph
from aima.search import compare_searchers

romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))

romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))

""" [Figure 4.9]
Eight possible states of the vacumm world
Each state is represented as
   *       "State of the left room"      "State of the right room"   "Room in which the agent
                                                                      is present"
1 - DDL     Dirty                         Dirty                       Left
2 - DDR     Dirty                         Dirty                       Right
3 - DCL     Dirty                         Clean                       Left
4 - DCR     Dirty                         Clean                       Right
5 - CDL     Clean                         Dirty                       Left
6 - CDR     Clean                         Dirty                       Right
7 - CCL     Clean                         Clean                       Left
8 - CCR     Clean                         Clean                       Right
"""
vacuum_world = Graph(dict(
    State_1=dict(Suck=['State_7', 'State_5'], Right=['State_2']),
    State_2=dict(Suck=['State_8', 'State_4'], Left=['State_2']),
    State_3=dict(Suck=['State_7'], Right=['State_4']),
    State_4=dict(Suck=['State_4', 'State_2'], Left=['State_3']),
    State_5=dict(Suck=['State_5', 'State_1'], Right=['State_6']),
    State_6=dict(Suck=['State_8'], Left=['State_5']),
    State_7=dict(Suck=['State_7', 'State_3'], Right=['State_8']),
    State_8=dict(Suck=['State_8', 'State_6'], Left=['State_7'])
))

""" [Figure 4.23]
One-dimensional state space Graph
"""
one_dim_state_space = Graph(dict(
    State_1=dict(Right='State_2'),
    State_2=dict(Right='State_3', Left='State_1'),
    State_3=dict(Right='State_4', Left='State_2'),
    State_4=dict(Right='State_5', Left='State_3'),
    State_5=dict(Right='State_6', Left='State_4'),
    State_6=dict(Left='State_5')
))
one_dim_state_space.least_costs = dict(
    State_1=8,
    State_2=9,
    State_3=2,
    State_4=2,
    State_5=4,
    State_6=3)

""" [Figure 6.1]
Principal states and territories of Australia
"""
australia_map = UndirectedGraph(dict(
    T=dict(),
    SA=dict(WA=1, NT=1, Q=1, NSW=1, V=1),
    NT=dict(WA=1, Q=1),
    NSW=dict(Q=1, V=1)))
australia_map.locations = dict(WA=(120, 24), NT=(135, 20), SA=(135, 30),
                               Q=(145, 20), NSW=(145, 32), T=(145, 42),
                               V=(145, 37))

def compare_graph_searchers():
    """Prints a table of search results."""
    compare_searchers(problems=[GraphProblem('Arad', 'Bucharest', romania_map),
                                GraphProblem('Oradea', 'Neamt', romania_map),
                                GraphProblem('Q', 'WA', australia_map)],
                      header=['Searcher', 'romania_map(Arad, Bucharest)',
                              'romania_map(Oradea, Neamt)', 'australia_map'])
