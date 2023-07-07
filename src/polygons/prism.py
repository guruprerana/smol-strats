from typing import List

from src.linpreds import Direction, DirectionSets
from src.polygons import HalfEdge, PolygonGridWorld, Vertex, det


def halfedge_to_prism(edge: HalfEdge, grid_factor=100) -> str:
    scaled_p1, scaled_p2 = (
        edge.start.mult_const(grid_factor),
        edge.end.mult_const(grid_factor),
    )

    d = scaled_p2 - scaled_p1
    return f"{d.x}*(y - {scaled_p2.y}) - {d.y}*(x - {scaled_p2.x}) >= 0"


def polygon_to_prism(edge: HalfEdge, grid_factor=100) -> str:
    return " & ".join(
        halfedge_to_prism(e, grid_factor) for e in edge.edges_in_polygon()
    )


def polygon_to_prism_with_actions(edge: HalfEdge, grid_factor=100) -> List[str]:
    res = []
    polygon_pred = polygon_to_prism(edge, grid_factor)
    dirs = DirectionSets[edge.actions] if edge.actions else []
    for dir in dirs:
        if dir == Direction.L:
            res.append(f"[] {polygon_pred} -> (x' = x - 1);")
        elif dir == Direction.R:
            res.append(f"[] {polygon_pred} -> (x' = x + 1);")
        elif dir == Direction.U:
            res.append(f"[] {polygon_pred} -> (y' = y + 1);")
        elif dir == Direction.D:
            res.append(f"[] {polygon_pred} -> (y' = y - 1);")
        else:
            raise ValueError

    return res


def polygon_grid_to_prism(
    gridw: PolygonGridWorld, filename: str, grid_factor=100, start_pt: Vertex = None
) -> None:
    if not start_pt:
        start_pt = Vertex(0, 0)
    with open(filename, "w") as f:
        f.writelines(
            (
                "mdp\n",
                "\n",
                "module grid\n",
                f"\tx : [0..{grid_factor*gridw.grid_size}] init {start_pt.x};\n",
                f"\ty : [0..{grid_factor*gridw.grid_size}] init {start_pt.y};\n",
                "\n",
            )
        )

        def write_polygon(edge: HalfEdge) -> None:
            action_strings = polygon_to_prism_with_actions(edge, grid_factor)
            action_strings.append("\n")
            action_strings = [f"\t{action}\n" for action in action_strings]

            f.writelines(action_strings)

        gridw.traverse_polygons(write_polygon)

        target_region = polygon_to_prism(gridw.target, grid_factor)
        f.writelines("endmodule\n\n")
        f.writelines((f'label "target" = {target_region};\n\n',))
