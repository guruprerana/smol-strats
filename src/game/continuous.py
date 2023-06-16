from src.polygons import HalfEdge, PolygonGridWorld, Vertex


class ContinuousReachabilityGridGame:
    def __init__(
        self, gridw: PolygonGridWorld, start_edge: HalfEdge, start_coord: Vertex
    ) -> None:
        self.gridw = gridw
        self.start_edge = start_edge
        self.current_edge = start_edge
        self.start_coord = start_coord
        self.current_coord = start_coord

        self.target_edges = set(gridw.target.edges_in_polygon())

    def restart(self):
        self.current_edge = self.start_edge
        self.current_coord = self.start_coord

    def step(self, direction: HalfEdge):
        assert self.current_coord == direction.start
        intersection_vertex = self.current_edge.intersects_edge(direction)
        intersection_edge = self.current_edge.next

        while not intersection_vertex and intersection_edge != self.current_edge:
            intersection_vertex = intersection_edge.intersects_edge(direction)
            intersection_edge = intersection_edge.next

        if not intersection_vertex:
            raise ValueError

        self.current_coord = intersection_vertex

        if intersection_edge in self.target_edges:
            return self.current_coord, None, True

        if not intersection_edge.opp:
            raise ValueError

        self.current_edge = intersection_edge.opp
        return self.current_coord, self.current_edge, False
