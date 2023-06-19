from src.polygons import HalfEdge, PolygonGridWorld, Vertex


class BasePolicy:
    def __init__(self, gridw: PolygonGridWorld) -> None:
        raise NotImplementedError

    def build(self) -> None:
        raise NotImplementedError

    def navigate(self, coord: Vertex, edge: HalfEdge) -> HalfEdge:
        raise NotImplementedError

    def restart(self) -> None:
        raise NotImplementedError
