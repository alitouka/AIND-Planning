from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        '''

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            '''
            loads = []
            # TODO create all load ground actions from the domain Load action
            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            return unloads

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """

        # Code copied from example_have_cake.py
        possible_actions = []
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
            if is_possible:
                possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # Code copied from example_have_cake.py
        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)
        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

def generate_propositions(relation:str, obj1:list, obj2:list, all_obj2:list):
    """
    Generates propositions which describe a fact that a relation holds for some objects but doesn't hold for
    all other objects.

    :param relation: a binary relation, such as 'In' or 'At'
    :param obj1:     a list of objects which can be the first argument of a relation, and for which a relation holds
    :param obj2:     a list of objects which can be the 2nd argument of a relation, and for which a relation holds
    :param all_obj2: a list of ALL objects of the same type as obj2.

    :return: two lists. The first contains propositions which are true, the second contains propositions which are false

     For example, if a plane P1 is at SFO airport, it cannot be in any other airports. So, if you call this function as
      generate_propositions('At', ['P1'], ['SFO'], ['SFO', 'JFK', 'ATL'])
     then it will return two lists:
      [expr('At(P1, SFO)')]
     and
      [expr('At(P1, JFK)')], [expr('At(P1, ATL)')],
    """
    pos = []
    neg = []

    for o1 in obj1:
        for o2 in obj2:
            pos.append(expr("{}({}, {})".format(relation, o1, o2)))

    for o1 in obj1:
        for o2 in set(all_obj2).difference(set(obj2)):
            neg.append(expr("{}({}, {})".format(relation, o1, o2)))

    return pos, neg

def append_propositions(relation:str, obj1:list, obj2:list, all_obj2:list, pos, neg):
    """
    Appends propositions generated by a generate_propositions method to corresponding lists

    :param pos: a list which positive propositions should be appended to
    :param neg: a list which negative propositions should be appended to

    All other parameters are passed to generate_propositions method. See their descriptions in documentation for this method
    """
    p, n = generate_propositions(relation, obj1, obj2, all_obj2)
    pos.extend(p)
    neg.extend(n)

def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = []
    neg = []

    # Cargos at airports
    append_propositions('At', ['C1'], ['SFO'], airports, pos, neg)
    append_propositions('At', ['C2'], ['JFK'], airports, pos, neg)
    append_propositions('At', ['C3'], ['ATL'], airports, pos, neg)

    # Planes at airports
    append_propositions('At', ['P1'], ['SFO'], airports, pos, neg)
    append_propositions('At', ['P2'], ['JFK'], airports, pos, neg)
    append_propositions('At', ['P3'], ['ATL'], airports, pos, neg)

    # None of the cargos are loaded on any plane
    append_propositions('In', cargos, [], planes, pos, neg)

    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = []
    neg = []

    # Cargos at airports
    append_propositions('At', ['C1'], ['SFO'], airports, pos, neg)
    append_propositions('At', ['C2'], ['JFK'], airports, pos, neg)
    append_propositions('At', ['C3'], ['ATL'], airports, pos, neg)
    append_propositions('At', ['C4'], ['ORD'], airports, pos, neg)

    # Planes at airports
    append_propositions('At', ['P1'], ['SFO'], airports, pos, neg)
    append_propositions('At', ['P2'], ['JFK'], airports, pos, neg)

    # None of the cargos are loaded on any plane
    append_propositions('In', cargos, [], planes, pos, neg)

    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

if __name__=="__main__":
    air_cargo_p3()