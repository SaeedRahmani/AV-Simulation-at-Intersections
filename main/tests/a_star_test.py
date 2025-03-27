import math
import unittest
from typing import Tuple, Iterable

from lib.a_star import AStar


def dummy_heuristic(node):
    # turns A* into Dijkstra
    return 0


class MyTestCase(unittest.TestCase):

    def build_networkx_graph(self):
        import networkx as nx

        # directed graph from Assignment 1 task 5
        g = nx.DiGraph()
        g.add_nodes_from(['Start', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Goal'])
        g.add_edges_from([
            ('Start', 'A', {'weight': 3}),
            ('Start', 'B', {'weight': 1}),
            ('Start', 'D', {'weight': 4}),
            ('A', 'C', {'weight': 2}),
            ('B', 'D', {'weight': 5}),
            ('B', 'E', {'weight': 6}),
            ('B', 'H', {'weight': 1}),
            ('C', 'D', {'weight': 2}),
            ('C', 'F', {'weight': 1}),
            ('C', 'Goal', {'weight': 9}),
            ('D', 'L', {'weight': 2}),
            ('E', 'J', {'weight': 4}),
            ('F', 'D', {'weight': 1}),
            ('G', 'K', {'weight': 3}),
            ('H', 'G', {'weight': 4}),
            ('H', 'I', {'weight': 6}),
            ('H', 'O', {'weight': 2}),
            ('I', 'J', {'weight': 5}),
            ('J', 'G', {'weight': 3}),
            ('J', 'Goal', {'weight': 3}),
            ('K', 'N', {'weight': 1}),
            ('M', 'K', {'weight': 1}),
            ('M', 'Goal', {'weight': 2}),
            ('N', 'M', {'weight': 2}),
            ('O', 'L', {'weight': 2}),
        ])

        return g

    def test_networkx_graph_dijkstra(self):
        # ------ SETUP THE TEST ------

        g = self.build_networkx_graph()

        def networkx_neighbor_function(node: str) -> Iterable[Tuple[float, str]]:
            # returns an iterable of tuples (edge_cost, node), just like the A* implementation wants
            return ((data['weight'], neighbor) for neighbor, data in g[node].items())

        # the node type in this case is plain string
        a_star: AStar[str] = AStar(neighbor_function=networkx_neighbor_function)

        def is_goal(node: str) -> bool:
            return node == 'Goal'

        # ------ RUN THE CODE ------

        # dummy heuristic turns A* into Dijkstra
        # run Dijkstra
        value, path = a_star.run(start='Start', is_goal_function=is_goal,
                                 heuristic_function=dummy_heuristic,
                                 debug=True)

        # ------ ASSERT RESULTS CORRECT ------

        # check that the value is 14
        self.assertEqual(14, value)

        # check that the expected path is returned
        self.assertEqual(['Start', 'A', 'C', 'Goal'], path)

        # check that the debug output says that the items are opened in the exact order that we expect Dijkstra to do
        self.assertEqual([
            ('Start', 0, 'Start'),
            ('B', 1, 'Start'),
            ('H', 2, 'B'),
            ('A', 3, 'Start'),
            ('D', 4, 'Start'),
            ('O', 4, 'H'),
            ('C', 5, 'A'),
            ('F', 6, 'C'),
            ('G', 6, 'H'),
            ('L', 6, 'D'),
            ('E', 7, 'B'),
            ('I', 8, 'H'),
            ('K', 9, 'G'),
            ('N', 10, 'K'),
            ('J', 11, 'E'),
            ('M', 12, 'N'),
            ('Goal', 14, 'C'),
        ], a_star.debug_data)

    def assert_path_correct(self, neighbor_function, path: list):
        for node, neighbor_actual in zip(path[:-1], path[1:]):
            neighbors_expected = [n for _, n in neighbor_function(node)]
            self.assertIn(neighbor_actual, neighbors_expected)

    def test_dynamically_generated_1d_grid(self):
        # ------ SETUP THE TEST ------

        NodeType = int

        def neighbor_function(node: NodeType) -> Iterable[Tuple[float, NodeType]]:
            for a in [-1, 1]:
                neighbor = node + a
                edge_cost = 1.

                yield edge_cost, neighbor

        a_star: AStar[NodeType] = AStar(neighbor_function=neighbor_function)

        start_node: NodeType = 0
        end_node: NodeType = 10

        def distance(node: NodeType) -> float:
            return abs(node - end_node)

        def is_goal_function(node: NodeType) -> bool:
            return node == end_node

        # ------ RUN THE CODE ------

        dijkstra_value, dijkstra_path = a_star.run(start=start_node,
                                                   is_goal_function=is_goal_function,
                                                   heuristic_function=dummy_heuristic,
                                                   debug=True)

        dijkstra_debug = a_star.debug_data

        astar_value, astar_path = a_star.run(start=start_node,
                                             is_goal_function=is_goal_function,
                                             heuristic_function=distance,
                                             debug=True)

        astar_debug = a_star.debug_data

        # ------ ASSERT RESULTS CORRECT ------

        # check value
        expected_value = distance(start_node)
        self.assertEqual(expected_value, dijkstra_value)
        self.assertEqual(expected_value, astar_value)

        # check path
        self.assert_path_correct(neighbor_function, dijkstra_path)
        self.assert_path_correct(neighbor_function, astar_path)

        # check that Dijkstra opens exactly 1 + 10 + 10 nodes (i.e. all in the range [-10, +10])
        self.assertEqual(21, len(dijkstra_debug))

        # check that A* opens exactly 1 + 10 nodes (i.e. only those in the range [0, +10])
        self.assertEqual(11, len(astar_debug))

        print("Dijkstra")
        print(dijkstra_debug)
        print()

        print("A*")
        print(astar_debug)
        print()

    def test_dynamically_generated_2d_grid(self):
        # ------ SETUP THE TEST ------

        NodeType = Tuple[int, int]

        def infinite_grid_neighbor_function(node: NodeType) -> Iterable[Tuple[float, NodeType]]:
            x, y = node
            for a, b in [[-1, 0], [0, -1], [1, 0], [0, 1]]:
                neighbor = x + a, y + b
                edge_cost = 1.

                yield edge_cost, neighbor

        a_star: AStar[NodeType] = AStar(neighbor_function=infinite_grid_neighbor_function)

        start_node: NodeType = (0, 0)
        end_node: NodeType = (10, 10)

        def euclidean_distance(node: NodeType) -> float:
            xa, ya = node
            xb, yb = end_node

            return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

        def manhattan_distance(node: NodeType) -> float:
            xa, ya = node
            xb, yb = end_node

            return abs(xa - xb) + abs(ya - yb)

        def is_goal_function(node: NodeType) -> bool:
            return node == end_node

        # ------ RUN THE CODE ------

        # run Dijkstra (dummy heuristic)
        dijkstra_value, dijkstra_path = a_star.run(start=start_node,
                                                   is_goal_function=is_goal_function,
                                                   heuristic_function=dummy_heuristic,
                                                   debug=True)

        dijkstra_debug = a_star.debug_data

        # run A* with Euclidean distance heuristic
        euclidean_value, euclidean_path = a_star.run(start=start_node,
                                                     is_goal_function=is_goal_function,
                                                     heuristic_function=euclidean_distance,
                                                     debug=True)
        euclidean_debug = a_star.debug_data

        # run A* with Manhattan distance heuristic
        manhattan_value, manhattan_path = a_star.run(start=start_node,
                                                     is_goal_function=is_goal_function,
                                                     heuristic_function=manhattan_distance,
                                                     debug=True)
        manhattan_debug = a_star.debug_data

        # ------ ASSERT RESULTS CORRECT ------

        # check value correct
        expected_value = manhattan_distance(start_node)
        self.assertEqual(expected_value, dijkstra_value)
        self.assertEqual(expected_value, euclidean_value)
        self.assertEqual(expected_value, manhattan_value)

        # check path correct
        self.assert_path_correct(infinite_grid_neighbor_function, dijkstra_path)
        self.assert_path_correct(infinite_grid_neighbor_function, euclidean_path)
        self.assert_path_correct(infinite_grid_neighbor_function, manhattan_path)

        # count the number of lines in the debug output
        dijkstra_n_openings = len(dijkstra_debug)
        euclidean_n_openings = len(euclidean_debug)
        manhattan_n_openings = len(manhattan_debug)

        # check euclidean strictly better than dijkstra
        self.assertLess(euclidean_n_openings, dijkstra_n_openings)

        # check manhattan strictly better than euclidean
        self.assertLess(manhattan_n_openings, euclidean_n_openings)

        print(f" Dijkstra: opened {dijkstra_n_openings} nodes")
        print(f"Euclidean: opened {euclidean_n_openings} nodes")
        print(f"Manhattan: opened {manhattan_n_openings} nodes")


if __name__ == '__main__':
    unittest.main()
