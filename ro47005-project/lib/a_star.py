import sys
from heapq import heappop, heappush
from io import StringIO
from typing import Callable, Iterable, List, TypeVar, Hashable, Tuple, Dict, Generic

# generic node type -> can be any type in practice; this is just for IDE autocompletion hints
TNode = TypeVar("TNode", bound=Hashable)


class AStar(Generic[TNode]):

    def __init__(self, neighbor_function: Callable[[TNode], Iterable[Tuple[float, TNode]]]):
        self.neighbor_function = neighbor_function

    def run(self, start: TNode, end: TNode,
            heuristic_function: Callable[[TNode, TNode], float],
            debug=False, debug_file=sys.stderr) -> Tuple[float, List[TNode]]:
        q: List[Tuple[float, float, TNode, TNode]] = [(0, 0, start, start)]  # G + H value, G value, node, predecessor

        # best predecessor dict
        pred_dict: Dict[TNode, Tuple[float, TNode]] = {}

        while q:
            gh, g, node, predecessor = heappop(q)

            if node in pred_dict and pred_dict[node][0] <= g:
                # we have been in this node before and with a shorter or the same path -> skip

                # TODO: if your heuristic is admissible,
                #  then the `pred_dict[node][0] <= g` check can be removed
                #  because it will be guaranteed!
                continue

            if debug:
                print(f"Opening {node} in distance {g}, with predecessor {predecessor}.", file=debug_file)

            # store value predecessor
            pred_dict[node] = g, predecessor

            if node == end:
                # we are done
                # reconstruct path
                path = [end]

                while node != start:
                    path.append(predecessor)
                    node, predecessor = predecessor, pred_dict[predecessor][1]

                path.reverse()
                return g, path

            # process neighbors
            for edge_value, neighbor in self.neighbor_function(node):
                neighbor_g = g + edge_value
                neighbor_gh = neighbor_g + heuristic_function(neighbor, end)

                heappush(q, (neighbor_gh, neighbor_g, neighbor, node))

        raise Exception("No solution found.")

    def run_with_debug(self, start: TNode, end: TNode,
                       heuristic_function: Callable[[TNode, TNode], float]) -> Tuple[float, List[TNode], str]:
        with StringIO() as debug_file:
            value, path = self.run(start=start, end=end, heuristic_function=heuristic_function,
                                   debug=True, debug_file=debug_file)
            debug_file.seek(0)

            return value, path, debug_file.read()
