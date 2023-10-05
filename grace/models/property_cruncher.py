import networkx as nx
import numpy as np

from grace.base import GraphAttrs, Properties


class EdgePropertyCruncher:
    """Calculator for edge attributes in graph as EDGE_PROPERTIES dataclass.

    Parameters
    ----------
    graph : nx.Graph
        Graph whose edges will be modified with additional edge properties.

    Notes
    -----
    - Normalisation for selected values = item / mean(all edge items in graph)
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

        self.mean_edge_length = None
        self.mean_triangle_area = None

        # TODO: finish up!
        self.x_mn = None
        self.y_mn = None
        self.x_mx = None
        self.y_mx = None

    def universal_edge_properties(self) -> None:
        """Property calculator for most properties.

        TODO: Include standardisation of the coordinates,
              and relative length to the edge length.

        Notes
        -----
        - Modifies the graph in-place.
        - Creates GraphAttrs.EDGE_PROPERTIES
        """

        for src, dst, edge in self.graph.edges(data=True):
            edge_attributes = {}

            src_coords = np.array(
                [
                    self.graph.nodes[src][GraphAttrs.NODE_X],
                    self.graph.nodes[src][GraphAttrs.NODE_Y],
                ]
            )
            dst_coords = np.array(
                [
                    self.graph.nodes[dst][GraphAttrs.NODE_X],
                    self.graph.nodes[dst][GraphAttrs.NODE_Y],
                ]
            )
            mid_coords = (src_coords + dst_coords) / 2.0

            # Store these raw positions:
            edge_attributes["src_pos_x_raw"] = src_coords[0]
            edge_attributes["src_pos_y_raw"] = src_coords[1]
            edge_attributes["dst_pos_x_raw"] = dst_coords[0]
            edge_attributes["dst_pos_y_raw"] = dst_coords[1]
            edge_attributes["mid_pos_x_raw"] = mid_coords[0]
            edge_attributes["mid_pos_y_raw"] = mid_coords[1]

            # edge length (raw):
            edge_length_raw = self.calculate_points_distance(
                src_coords, dst_coords
            )
            edge_attributes["edge_length_raw"] = edge_length_raw

            # edge orientation:
            edge_orientation = self.calculate_angle_with_vertical(
                src_coords, dst_coords
            )
            edge_attributes["edge_orientation_radians"] = edge_orientation[0]
            edge_attributes["edge_orientation_degrees"] = edge_orientation[1]

            # Identify the left & right partner node:
            neighbours = {
                "west": float("inf"),
                "east": float("inf"),
            }

            neighbours_src = set(self.graph.neighbors(src))
            neighbours_dst = set(self.graph.neighbors(dst))

            # Find the intersection (common neighbors) between the two sets
            common_neighbours = neighbours_src.intersection(neighbours_dst)
            assert len(common_neighbours) > 0

            # It can certainly happen that a certain edge forms more than one
            # triangle on each east & west side - unwrap neighbours & choose:
            for neigh in common_neighbours:
                new_coords = np.array(
                    [
                        self.graph.nodes[neigh][GraphAttrs.NODE_X],
                        self.graph.nodes[neigh][GraphAttrs.NODE_Y],
                    ]
                )
                neighbour_position = self.point_position_relative_to_edge(
                    src_coords, dst_coords, new_coords
                )
                mid_to_point_length_raw = self.calculate_points_distance(
                    mid_coords, new_coords
                )
                mid_to_point_orient_raw = self.calculate_angle_with_vertical(
                    mid_coords, new_coords
                )
                triangle_area = self.calculate_triangle_area(
                    src_coords, dst_coords, new_coords
                )

                if neighbour_position == "west":
                    if mid_to_point_length_raw < neighbours["west"]:
                        neighbours["west"] = mid_to_point_length_raw

                        edge_attributes["west_pos_x_raw"] = new_coords[0]
                        edge_attributes["west_pos_y_raw"] = new_coords[1]

                        edge_attributes[
                            "west_to_mid_length_raw"
                        ] = mid_to_point_length_raw
                        edge_attributes[
                            "west_to_mid_orient_raw"
                        ] = mid_to_point_orient_raw[0]
                        edge_attributes[
                            "west_triangle_area_raw"
                        ] = triangle_area

                elif neighbour_position == "east":
                    if mid_to_point_length_raw < neighbours["east"]:
                        neighbours["east"] = mid_to_point_length_raw

                        edge_attributes["east_pos_x_raw"] = new_coords[0]
                        edge_attributes["east_pos_y_raw"] = new_coords[1]

                        edge_attributes[
                            "east_to_mid_length_raw"
                        ] = mid_to_point_length_raw
                        edge_attributes[
                            "east_to_mid_orient_raw"
                        ] = mid_to_point_orient_raw[0]
                        edge_attributes[
                            "east_triangle_area_raw"
                        ] = triangle_area

                else:
                    raise ValueError("Node cannot be collinear to an edge.")

            # In case you only have 1 formed triangle:
            if neighbours["west"] == float("inf"):
                edge_attributes["west_pos_x_raw"] = mid_coords[0]
                edge_attributes["west_pos_y_raw"] = mid_coords[1]

                edge_attributes["west_to_mid_length_raw"] = 0.0
                edge_attributes["west_to_mid_orient_raw"] = 0.0
                edge_attributes["west_triangle_area_raw"] = 0.0

            if neighbours["east"] == float("inf"):
                edge_attributes["east_pos_x_raw"] = mid_coords[0]
                edge_attributes["east_pos_y_raw"] = mid_coords[1]

                edge_attributes["east_to_mid_length_raw"] = 0.0
                edge_attributes["east_to_mid_orient_raw"] = 0.0
                edge_attributes["east_triangle_area_raw"] = 0.0

            # Done - create the 'Properties' object to store attributes:
            edge[GraphAttrs.EDGE_PROPERTIES] = Properties(
                properties_dict=edge_attributes
            )

    def find_average_measure_values(self) -> None:
        real_edge_lengths = []
        real_triangle_areas = []

        for _, _, edge in self.graph.edges(data=True):
            dictionary = edge[GraphAttrs.EDGE_PROPERTIES].properties_dict
            real_edge_lengths.append(dictionary["edge_length_raw"])
            real_triangle_areas.append(dictionary["west_triangle_area_raw"])
            real_triangle_areas.append(dictionary["east_triangle_area_raw"])

        self.mean_edge_length = np.mean(real_edge_lengths)
        self.mean_triangle_area = np.mean(real_triangle_areas)

    def add_normalised_edge_attributes(self) -> None:
        for _, _, edge in self.graph.edges(data=True):
            dictionary = edge[GraphAttrs.EDGE_PROPERTIES].properties_dict

            # Add new normalised key-value pairs:
            dictionary["edge_length_nrm"] = (
                dictionary["edge_length_raw"] / self.mean_edge_length
            )

            dictionary["west_to_mid_length_nrm"] = (
                dictionary["west_to_mid_length_raw"] / self.mean_edge_length
            )
            dictionary["east_to_mid_length_nrm"] = (
                dictionary["east_to_mid_length_raw"] / self.mean_edge_length
            )

            dictionary["west_triangle_area_nrm"] = (
                dictionary["west_triangle_area_raw"] / self.mean_triangle_area
            )
            dictionary["east_triangle_area_nrm"] = (
                dictionary["east_triangle_area_raw"] / self.mean_triangle_area
            )

            edge[GraphAttrs.EDGE_PROPERTIES] = Properties(
                properties_dict=dictionary
            )

    def normalise_coordinates(self) -> None:
        pass

    def process(self) -> nx.Graph:
        self.universal_edge_properties()
        self.find_average_measure_values()
        self.add_normalised_edge_attributes()
        return self.graph

    @staticmethod
    def calculate_points_distance(src_coords, dst_coords) -> float:
        return np.linalg.norm(src_coords - dst_coords)

    @staticmethod
    def calculate_angle_with_vertical(
        src_point, dst_point
    ) -> tuple[float, float]:
        """Calculate angle between line & vertical plane (in radians)."""

        # Calculate the midpoint of the line
        x_src, y_src = src_point
        x_dst, y_dst = dst_point

        x_mid = (x_src + x_dst) / 2.0
        y_mid = (y_src + y_dst) / 2.0

        # Calculate the angle using arctan2, measured from the vertical axis
        angle_rad = np.arctan2(x_mid, y_mid)

        # Convert the angle from radians to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_rad, angle_deg

    @staticmethod
    def calculate_triangle_area(src_coords, dst_coords, ver_coords) -> float:
        """Calculate the area of a triangle defined by three sets of coordinates."""
        x1, y1 = src_coords
        x2, y2 = dst_coords
        x3, y3 = ver_coords

        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        return area

    @staticmethod
    def point_position_relative_to_edge(src_node, dst_node, new_node):
        """
        Determine whether a new point is to the left or to the right
        of an edge line defined by its source and destination nodes.

        Returns: str
            A string indicating the relative position: 'left', 'right'
            or 'collinear' if the point is collinear with the edge line.
        """
        x_src, y_src = src_node
        x_dst, y_dst = dst_node
        x_new, y_new = new_node

        # Vector representing the edge line segment
        edge_vector = np.array([x_dst - x_src, y_dst - y_src])

        # Vector from the source node to the new point
        new_point_vector = np.array([x_new - x_src, y_new - y_src])

        # Calculate the cross product
        cross_product = np.cross(edge_vector, new_point_vector)

        if cross_product > 0:
            return "west"
        elif cross_product < 0:
            return "east"
        else:
            return "collinear"
