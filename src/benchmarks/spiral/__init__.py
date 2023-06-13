from src.polygons.prism import polygon_grid_to_prism
from src.backward_reachability import BackwardReachabilityTree
from src.linpreds import Direction, actions_from_directions
from src.polygons import HalfEdge, Vertex, simple_rectangular_polygon_gridw


def main():
    gridw = simple_rectangular_polygon_gridw(size=27)
    gridw.root.prev.add_intersection_edge(HalfEdge(Vertex(0, 21), Vertex(27, 21)))
    gridw.root.prev.add_intersection_edge(HalfEdge(Vertex(0, 11), Vertex(27, 11)))
    gridw.root.prev.add_intersection_edge(HalfEdge(Vertex(0, 10), Vertex(27, 10)))
    gridw.root.navigate("nfn").add_vertex(Vertex(10, 11)).add_vertex(
        Vertex(13, 11)
    ).add_vertex(Vertex(14, 11)).add_vertex(Vertex(17, 11))
    gridw.root.navigate("nn").add_vertex(Vertex(10, 10)).add_vertex(
        Vertex(13, 10)
    ).add_vertex(Vertex(14, 10)).add_vertex(Vertex(17, 10))
    gridw.root = gridw.root.add_vertex(Vertex(24, 0)).add_vertex(Vertex(23, 0))

    gridw.root.navigate("pbbp").add_diagonal(Vertex(10, 11))
    gridw.root.navigate("pbb").assign_action_polygon(
        actions_from_directions([Direction.D])
    )
    gridw.root.navigate("pbbbp").add_diagonal(Vertex(17, 11))
    gridw.root.navigate("pbbb").assign_action_polygon(
        actions_from_directions([Direction.L])
    )
    gridw.root.navigate("pbbbb").assign_action_polygon(
        actions_from_directions([Direction.U])
    )
    gridw.root.navigate("ppo").add_diagonal(Vertex(10, 11))
    gridw.root.navigate("ppo").assign_action_polygon(
        actions_from_directions([Direction.D])
    )
    gridw.root.prev.add_diagonal(Vertex(10, 10))
    gridw.root.prev.opp.assign_action_polygon(actions_from_directions([Direction.D]))
    gridw.root.navigate("bpbppop").add_diagonal(Vertex(14, 11))
    gridw.root.navigate("bpbo").add_diagonal(Vertex(13, 11))
    gridw.target = gridw.root.navigate("pppo")
    gridw.root.navigate("ppppo").add_diagonal(Vertex(17, 11))
    gridw.root.navigate("ppppof").assign_action_polygon(
        actions_from_directions([Direction.U])
    )
    gridw.root.navigate("nn").add_diagonal(Vertex(17, 10))
    gridw.root.navigate("nnf").assign_action_polygon(
        actions_from_directions([Direction.U])
    )
    gridw.root.navigate("n").add_diagonal(Vertex(14, 10))
    gridw.root.navigate("nf").assign_action_polygon(
        actions_from_directions([Direction.R])
    )
    gridw.root.add_diagonal(Vertex(13, 10))
    gridw.root.navigate("f").assign_action_polygon(
        actions_from_directions([Direction.U])
    )
    gridw.root.assign_action_polygon(actions_from_directions([Direction.R]))

    gridw.draw("benchmarks/spiral/polygon-grid.png")

    btree = BackwardReachabilityTree(gridw)
    btree.construct_tree(max_depth=100)
    # btree.print()
    btree.draw(filename="benchmarks/spiral/backward-graph.png")

    polygon_grid_to_prism(gridw, "benchmarks/spiral/spiral.prism", 100)

    return gridw, btree
