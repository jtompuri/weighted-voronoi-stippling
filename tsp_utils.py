#!/usr/bin/env python3
"""Utility functions for working with TSP files generated from stippling.

This module provides comprehensive utilities for analyzing, visualizing, and
processing TSP (Traveling Salesman Problem) files created from stippling
algorithms. It includes functions for reading TSP files, calculating tour
distances, generating nearest-neighbor solutions, and exporting data for
various TSP solvers.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os


def read_tsp_file(filepath):
    """Read a TSP file and return the coordinates.

    Parses a TSP file in standard format and extracts node coordinates
    from the NODE_COORD_SECTION. The file is expected to follow the
    TSPLIB format specification.

    Args:
        filepath (str): Path to the TSP file to read.

    Returns:
        np.ndarray: Array of shape (n_nodes, 2) containing (x, y)
            coordinates for each node.
    """
    coordinates = []

    with open(filepath, 'r') as f:
        in_coord_section = False
        for line in f:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            elif line == "EOF":
                break
            elif in_coord_section and line:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: node_id x y (node_id not used in coordinates)
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates.append((x, y))

    return np.array(coordinates)


def calculate_tour_distance(coordinates, tour):
    """Calculate the total Euclidean distance of a tour.

    Computes the sum of Euclidean distances between consecutive nodes
    in the tour, including the return distance from the last node back
    to the first node to complete the cycle.

    Args:
        coordinates (np.ndarray): Array of shape (n_nodes, 2) containing
            (x, y) coordinates for each node.
        tour (list): List of node indices representing the tour order.

    Returns:
        float: Total distance of the tour.
    """
    total_distance = 0.0
    n = len(tour)

    for i in range(n):
        start = coordinates[tour[i]]
        end = coordinates[tour[(i + 1) % n]]
        distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        total_distance += distance

    return total_distance


def nearest_neighbor_tour(coordinates):
    """Generate a tour using the nearest neighbor heuristic.

    Implements the nearest neighbor algorithm for TSP, which starts at
    node 0 and repeatedly visits the closest unvisited node until all
    nodes have been visited.

    Args:
        coordinates (np.ndarray): Array of shape (n_nodes, 2) containing
            (x, y) coordinates for each node.

    Returns:
        list: Tour as a list of node indices representing the order
            to visit nodes.

    Note:
        This is a greedy heuristic that does not guarantee optimal
        solutions but provides reasonable results quickly.
    """
    n = len(coordinates)
    unvisited = set(range(1, n))
    tour = [0]  # Start at node 0
    current = 0

    while unvisited:
        # Find nearest unvisited node
        distances = [np.linalg.norm(coordinates[current] - coordinates[x])
                     for x in unvisited]
        nearest_idx = np.argmin(distances)
        nearest = list(unvisited)[nearest_idx]

        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return tour


def visualize_tsp_tour(coordinates, tour=None, output_file=None):
    """Visualize TSP coordinates and optionally a tour path.

    Creates a matplotlib plot showing the TSP nodes as scatter points
    and optionally draws the tour path connecting the nodes in order.

    Args:
        coordinates (np.ndarray): Array of shape (n_nodes, 2) containing
            (x, y) coordinates for each node.
        tour (list, optional): List of node indices representing the tour
            order. If None, only points are plotted. Defaults to None.
        output_file (str, optional): Path to save the plot image. If None,
            the plot is displayed interactively. Defaults to None.
    """
    plt.figure(figsize=(10, 8))

    # Plot points
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=20, zorder=3)

    # Plot tour if provided
    if tour is not None:
        tour_coords = coordinates[tour + [tour[0]]]  # Close the loop
        plt.plot(tour_coords[:, 0], tour_coords[:, 1],
                 'b-', linewidth=1, alpha=0.7, zorder=2)

        # Calculate and display tour distance
        distance = calculate_tour_distance(coordinates, tour)
        plt.title(f'TSP Tour (Distance: {distance:.2f})')
    else:
        plt.title('TSP Points')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Tour visualization saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def generate_distance_matrix(coordinates):
    """Generate a symmetric distance matrix for the TSP instance.

    Computes the Euclidean distance between every pair of nodes and
    returns it as a symmetric matrix suitable for TSP solvers that
    require distance matrices.

    Args:
        coordinates (np.ndarray): Array of shape (n_nodes, 2) containing
            (x, y) coordinates for each node.

    Returns:
        np.ndarray: Symmetric distance matrix of shape (n_nodes, n_nodes)
            where element [i, j] is the Euclidean distance between
            nodes i and j.
    """
    return squareform(pdist(coordinates))


def export_for_concorde(tsp_file, output_file=None):
    """Export TSP file in format suitable for Concorde TSP solver.

    Reads a TSP file and converts it to the exact format expected by
    the Concorde TSP solver, ensuring compatibility with this high-
    performance solver.

    Args:
        tsp_file (str): Path to the input TSP file.
        output_file (str, optional): Path for the output file. If None,
            generates a filename by appending '_concorde' to the input
            filename. Defaults to None.

    Returns:
        str: Path to the generated Concorde-format TSP file.
    """
    coordinates = read_tsp_file(tsp_file)

    if output_file is None:
        base_name = os.path.splitext(tsp_file)[0]
        output_file = f"{base_name}_concorde.tsp"

    with open(output_file, 'w') as f:
        f.write(f"NAME: {os.path.basename(tsp_file)}\n")
        f.write("TYPE: TSP\n")
        f.write("COMMENT: Generated from stippling for Concorde solver\n")
        f.write(f"DIMENSION: {len(coordinates)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")

        for i, (x, y) in enumerate(coordinates, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")

        f.write("EOF\n")

    print(f"Concorde-format TSP file saved to: {output_file}")
    return output_file


def analyze_tsp_instance(tsp_file):
    """Analyze a TSP instance and provide comprehensive statistics.

    Loads a TSP file and computes various statistics about the instance
    including coordinate ranges, distance statistics, and provides a
    baseline solution using the nearest neighbor heuristic.

    Args:
        tsp_file (str): Path to the TSP file to analyze.

    Returns:
        tuple: A tuple containing:
            - coordinates (np.ndarray): Node coordinates
            - nn_tour (list): Nearest neighbor tour solution
    """
    coordinates = read_tsp_file(tsp_file)
    n = len(coordinates)

    # Calculate basic statistics
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    print(f"TSP Instance Analysis: {os.path.basename(tsp_file)}")
    print("=" * 50)
    print(f"Number of cities: {n}")
    print(f"X coordinate range: {x_coords.min():.2f} to "
          f"{x_coords.max():.2f}")
    print(f"Y coordinate range: {y_coords.min():.2f} to "
          f"{y_coords.max():.2f}")

    # Distance statistics
    distances = pdist(coordinates)
    print(f"Min distance between cities: {distances.min():.2f}")
    print(f"Max distance between cities: {distances.max():.2f}")
    print(f"Average distance: {distances.mean():.2f}")

    # Simple tour estimation
    nn_tour = nearest_neighbor_tour(coordinates)
    nn_distance = calculate_tour_distance(coordinates, nn_tour)
    print(f"Nearest neighbor tour distance: {nn_distance:.2f}")

    return coordinates, nn_tour


def main():
    """Command-line interface for TSP utilities.

    Provides a command-line interface to access all TSP utility functions
    including analysis, visualization, nearest neighbor solving, and
    export capabilities.
    """
    parser = argparse.ArgumentParser(description="TSP utilities for stippling")
    parser.add_argument("tsp_file", help="Path to TSP file")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze the TSP instance")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the TSP points")
    parser.add_argument("--nearest-neighbor", action="store_true",
                        help="Solve with nearest neighbor and visualize")
    parser.add_argument("--export-concorde", action="store_true",
                        help="Export for Concorde TSP solver")
    parser.add_argument(
        "--output", help="Output file for visualization or export")

    args = parser.parse_args()

    if not os.path.exists(args.tsp_file):
        print(f"Error: TSP file '{args.tsp_file}' not found")
        return

    if args.analyze:
        coordinates, nn_tour = analyze_tsp_instance(args.tsp_file)

        if args.visualize:
            if args.nearest_neighbor:
                visualize_tsp_tour(coordinates, nn_tour, args.output)
            else:
                visualize_tsp_tour(coordinates, None, args.output)

    elif args.visualize:
        coordinates = read_tsp_file(args.tsp_file)
        if args.nearest_neighbor:
            nn_tour = nearest_neighbor_tour(coordinates)
            visualize_tsp_tour(coordinates, nn_tour, args.output)
        else:
            visualize_tsp_tour(coordinates, None, args.output)

    elif args.export_concorde:
        export_for_concorde(args.tsp_file, args.output)

    else:
        # Default: just analyze
        analyze_tsp_instance(args.tsp_file)


if __name__ == "__main__":
    main()
