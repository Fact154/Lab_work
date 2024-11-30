'''
Задача о положении трубных секций
1. Нужно реализовать структуру графа( с операцией добавления и удаления ребер с вершинами).
Каждая вершина графа должна содержать название(метку) и координаты в пространстве, каждое ребро может содержать метку
2. Добавить реализовать метод/функцию перемещения по графу(задается начальная точка – одна из вершин построенного графа)
 и дистанция на которую нужно переместиться, в качестве результата должны выводиться примерные координаты
(сектор с ожидаемым местоположением) размер сектора зависит от параметра ожидаемой точности
Пример:
vertexA = Vertex(‘A’, 0, 0, 0)
vertexB = Vertex(‘B’, 20, 0, 0)
vertexC = Vertex(‘C’, 12, 12, 12)

edgeAB = Edge(’20’)
edgeBC = Edge(’30’)

graph = Graph()

graph.addVertexes(vertexA, vertexB, vertexC)
graph.addEdges(
	(edgeAB, ‘A’, `B’),
	(edgeBC, ‘B,’ ‘C’),
)

graph.findRoute(‘A’, 35, 0.2) # результат ожидаемый -> 	[(13.5, 9.5, 9.6), (13.7, 9.7, 9.6)] )
#  [(13.5, 9.5, 9.6), (13.7, 9.7, 9.6)] – это координаты ожидаемого сектора с учетом точности 0.2 –
по максимальной метрике, в целом, можно сделать и чтоб точность была по Евклидовой.
Первая точка это нижняя левая вершина прямоугольника, а вторая правая верхняя
'''


import math
from typing import Dict, List, Tuple

# Класс вершины графа
class Vertex:
    def __init__(self, label: str, x: float, y: float, z: float) -> None:
        self.label: str = label  # Название вершины
        self.x: float = x       # Координата X
        self.y: float = y       # Координата Y
        self.z: float = z       # Координата Z

    def coordinates(self) -> Tuple[float, float, float]:
        # Возвращает координаты вершины в виде кортежа
        return self.x, self.y, self.z

# Класс ребра графа
class Edge:
    def __init__(self, label: str) -> None:
        self.label: str = label  # Название ребра (например, длина)

# Класс графа
class Graph:
    def __init__(self) -> None:
        self.vertices: Dict[str, Vertex] = {}  # Словарь вершин: {метка вершины -> объект Vertex}
        self.edges: Dict[Tuple[str, str], Edge] = {}  # Словарь рёбер: {(метка_начала, метка_конца) -> объект Edge}

    # Добавление одной вершины
    def add_vertex(self, vertex: Vertex) -> None:
        self.vertices[vertex.label] = vertex

    # Добавление нескольких вершин
    def add_vertices(self, *vertices: Vertex) -> None:
        for vertex in vertices:
            self.add_vertex(vertex)

    # Добавление одного ребра
    def add_edge(self, edge: Edge, from_label: str, to_label: str) -> None:
        # Проверяем, что обе вершины существуют в графе
        if from_label not in self.vertices or to_label not in self.vertices:
            raise ValueError("Обе вершины должны существовать в графе.")
        # Добавляем ребро в словарь
        self.edges[(from_label, to_label)] = edge

    # Добавление нескольких рёбер
    def add_edges(self, *edges: Tuple[Edge, str, str]) -> None:
        for edge, from_label, to_label in edges:
            self.add_edge(edge, from_label, to_label)

    # Метод для нахождения примерного местоположения на графе
    def find_route(self, start_label: str, distance: float, precision: float) -> Tuple[str, List[Tuple[float, float, float]]]:
        # Проверяем, что начальная вершина существует
        if start_label not in self.vertices:
            raise ValueError("Начальной вершины в графе не существует.")

        current_label: str = start_label    # Метка текущей вершины
        remaining_distance: float = distance # Дистанция, которую нужно пройти

        # Перебираем рёбра графа
        for (from_label, to_label), edge in self.edges.items():
            if from_label == current_label:
                # Получаем начальную и конечную вершины текущего ребра
                from_vertex: Vertex = self.vertices[from_label]
                to_vertex: Vertex = self.vertices[to_label]
                # Рассчитываем длину ребра
                edge_length: float = self._distance(from_vertex, to_vertex)

                if remaining_distance <= edge_length:
                    # Если оставшаяся дистанция помещается на текущем ребре
                    # Вычислим пропорцию пути на ребре
                    ratio: float = remaining_distance / edge_length
                    # Рассчитываем примерные координаты с учетом точности
                    approximate_coords: List[Tuple[float, float, float]] = self._interpolate_coordinates(from_vertex, to_vertex, ratio, precision)
                    # Округляем координаты до 3 знаков
                    rounded_coords = [(round(x, 3), round(y, 3), round(z, 3)) for x, y, z in approximate_coords]
                    # Возвращаем метку ребра и координаты сектора
                    return edge.label, rounded_coords
                else:
                    # Если расстояние больше длины ребра, переходим к следующему ребру
                    remaining_distance -= edge_length
                    current_label = to_label

        # Если расстояние превышает длину всех рёбер
        raise ValueError("Расстояние больше, чем максимальная длина пути в графе.")

    # Метод для вычисления расстояния между двумя вершинами (Евклидово расстояние)
    @staticmethod
    def _distance(vertex1: Vertex, vertex2: Vertex) -> float:
        return math.sqrt(
            (vertex1.x - vertex2.x) ** 2 +
            (vertex1.y - vertex2.y) ** 2 +
            (vertex1.z - vertex2.z) ** 2
        )

    # Метод для линейной интерполяции координат между двумя вершинами
    @staticmethod
    def _interpolate_coordinates(v1: Vertex, v2: Vertex, ratio: float, precision: float) -> List[Tuple[float, float, float]]:
        # Рассчитываем координаты точки на заданной пропорции пути
        x: float = v1.x + ratio * (v2.x - v1.x)
        y: float = v1.y + ratio * (v2.y - v1.y)
        z: float = v1.z + ratio * (v2.z - v1.z)

        # Добавляем точность, чтобы создать сектор (прямоугольник вокруг точки)
        offset: float = precision
        return [
            (x - offset, y - offset, z - offset),  # Нижняя левая точка сектора
            (x + offset, y + offset, z + offset)  # Верхняя правая точка сектора
        ]

# Создаем вершины с их координатами
vertexA = Vertex('A', 0, 0, 0)
vertexB = Vertex('B', 20, 0, 0)
vertexC = Vertex('C', 12, 12, 12)
vertexD = Vertex('D', 12, 12, 20)

# Создаем рёбра с их метками
edgeAB = Edge('10')
edgeBC = Edge('20')
edgeCD = Edge('30')

# Создаем граф
graph = Graph()

# Добавляем вершины в граф
graph.add_vertices(vertexA, vertexB, vertexC, vertexD)

# Добавляем рёбра в граф
graph.add_edges(
    (edgeAB, 'A', 'B'),
    (edgeBC, 'B', 'C'),
    (edgeCD, 'C', 'D')
)

# Ищем примерное местоположение на графе
# Старт из вершины 'A', проходим расстояние 18, точность = 0.2
result = graph.find_route('A', 21, 0.2)

# Вывод результата
print(result)
