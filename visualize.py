"""Visualization utilities for TSP tours from stippling results.

This module provides comprehensive utilities for loading, processing, and
visualizing Traveling Salesman Problem (TSP) tour solutions generated from
stippling algorithms. It supports parsing standard TSP file formats and
tour files from various TSP solvers, with specialized support for Linkern
solver output.

The module enables visualization of stipple point distributions and their
corresponding optimal tour paths, making it useful for analyzing the quality
of stippling algorithms and TSP solver performance. The visualization
functions handle coordinate system transformations to ensure proper
orientation when displaying results.

Key Features:
    - Parse standard TSPLIB format files generated by stippling algorithms
    - Load tour solutions from Linkern and other TSP solvers
    - Visualize tour paths with top-left origin coordinate system
    - Support for multiple visualization modes: points-only, lines-only,
      or combined
    - Automatic defaults and flexible command-line interface
    - Smart TSP file detection when no path is specified

Supported File Formats:
    - TSP files: Standard TSPLIB format with NODE_COORD_SECTION
    - Tour files: Linkern solver output format with node sequences

Coordinate System:
    The module uses a top-left origin coordinate system that matches
    standard image coordinates (x increases right, y increases down).
    This eliminates the need for coordinate flipping and makes the
    visualization coordinates directly match the original image space.

Command line usage:
    $ python visualize.py stipplings/tsp/image_5000.tsp
    $ python visualize.py stipplings/tsp/image_5000.tsp \\
        --tour-path visualizations/tour/image_5000.tour --lines-only
    $ python visualize.py --tsp-path stipplings/tsp/image_5000.tsp \\
        --tour-path tour.tour --show-points --output tour.png

Dependencies:
    - matplotlib: For plotting and visualization
    - Standard library: File I/O and data processing

Note:
    This module is specifically designed to work with TSP files generated
    by the weighted Voronoi stippling algorithms in this package. While
    it may work with other TSP files, optimal results are achieved when
    used with the stippling workflow.

Authors:
    Visualization utilities for stippling and TSP analysis.
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def load_tsp_coordinates(tsp_path: str) -> List[Tuple[float, float]]:
    """Load coordinates from a TSP file format.

    Parses a TSP file and extracts the node coordinates from the
    NODE_COORD_SECTION. The TSP file format is expected to contain
    a header followed by coordinate data.

    Args:
        tsp_path (str): Path to the TSP file containing node coordinates.

    Returns:
        list: List of tuples containing (x, y) coordinate pairs as floats.
            Each tuple represents one node's position.

    Raises:
        FileNotFoundError: If the TSP file doesn't exist.
        ValueError: If the TSP file format is invalid or contains no
                   coordinates.
    """
    if not os.path.exists(tsp_path):
        raise FileNotFoundError(f"TSP file not found: {tsp_path}")

    coords = []
    with open(tsp_path, 'r') as f:
        in_section = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if line == "EOF":
                break
            if in_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords.append((float(parts[1]), float(parts[2])))
                    except ValueError:
                        print(f"Warning: Skipping invalid coordinate line: "
                              f"{line}")
                        continue

    if not coords:
        raise ValueError(f"No valid coordinates found in TSP file: {tsp_path}")

    return coords


def load_linkern_tour(tour_path: str) -> List[int]:
    """Load a tour solution from a Linkern TSP solver output file.

    Parses the tour file generated by Linkern TSP solver and extracts
    the sequence of nodes that represents the optimal tour path.

    Args:
        tour_path (str): Path to the tour file containing the TSP solution.

    Returns:
        list: List of integers representing the tour sequence. Each integer
            is a 0-based node index indicating the order to visit nodes.

    Raises:
        FileNotFoundError: If the tour file doesn't exist.
        ValueError: If no tour data is found in the file or format is invalid.
    """
    if not os.path.exists(tour_path):
        raise FileNotFoundError(f"Tour file not found: {tour_path}")

    tour = []
    with open(tour_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip first line (node count)
        if not lines:
            raise ValueError(f"No tour data found in file: {tour_path}")

        try:
            current = int(lines[0].split()[0])
            tour.append(current)
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    next_node = int(parts[1])
                    tour.append(next_node)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid tour file format: {tour_path}") from e

    if not tour:
        raise ValueError(f"No valid tour data found in file: {tour_path}")

    return tour


def plot_points_only(coords: List[Tuple[float, float]],
                     output_file: Optional[str] = None,
                     point_size: float = 1.0) -> None:
    """Plot only the stipple points without connecting tour lines.

    Creates a visualization showing just the stipple points as dots,
    useful for analyzing point distribution without tour optimization.
    Uses top-left origin coordinate system matching image coordinates.

    Args:
        coords: List of (x, y) coordinate tuples for each node.
        output_file: Path to save the plot. If None, displays interactively.
        point_size: Size of the points in the plot.
    """
    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=point_size, c='black', marker='.')

    # Set origin to top-left corner (like image coordinates)
    plt.gca().invert_yaxis()

    # Hide axes and frame
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Points visualization saved to: {output_file}")
        plt.close()
    else:
        plt.show()


def plot_tour(coords: List[Tuple[float, float]], tour: List[int],
              output_file: Optional[str] = None,
              line_width: float = 2.0,
              line_color: str = 'black',
              show_points: bool = False,
              point_size: float = 1.0) -> None:
    """Plot a TSP tour with top-left origin coordinate system.

    Creates a visualization of the TSP tour path by connecting the nodes
    in the order specified by the tour. Uses top-left origin coordinate
    system that matches image coordinates, eliminating the need for flipping.

    Args:
        coords: List of (x, y) coordinate tuples for each node.
        tour: List of node indices representing the tour order.
        output_file: Path to save the plot. If None, displays interactively.
        line_width: Width of the tour line.
        line_color: Color of the tour line.
        show_points: Whether to show the stipple points on top of the lines.
        point_size: Size of points if show_points is True.

    Raises:
        ValueError: If tour contains invalid node indices.
    """
    # Validate tour indices
    max_coord_index = len(coords) - 1
    for node_idx in tour:
        if node_idx < 0 or node_idx > max_coord_index:
            raise ValueError(f"Tour contains invalid node index {node_idx}. "
                             f"Valid range: 0-{max_coord_index}")

    # Extract coordinates and close the tour loop
    x = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
    y = [coords[i][1] for i in tour] + [coords[tour[0]][1]]

    # Plot tour line
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color=line_color, linewidth=line_width,
             solid_joinstyle='miter', solid_capstyle='butt')

    # Optionally add points on top of the lines
    if show_points:
        point_x = [coord[0] for coord in coords]
        point_y = [coord[1] for coord in coords]
        plt.scatter(point_x, point_y, s=point_size, c='red', marker='o',
                    zorder=3)  # zorder=3 puts points on top

    # Set origin to top-left corner (like image coordinates)
    plt.gca().invert_yaxis()

    # Hide axes and frame
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Tour visualization saved to: {output_file}")
        plt.close()
    else:
        plt.show()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize TSP tours from stippling results"
    )
    parser.add_argument("tsp_path", nargs='?',
                        help="Path to TSP file containing coordinates")
    parser.add_argument("--tsp-path", "--tsp", dest="tsp_path_alt",
                        help="Alternative way to specify TSP file path")
    parser.add_argument("--tour-path", "--tour",
                        help="Path to tour file containing TSP solution "
                             "(optional, if not provided, shows points only)")
    parser.add_argument("--output", "-o",
                        help="Save plot to file instead of displaying "
                             "(suggested: visualizations/png/filename.png)")
    parser.add_argument("--points-only", action="store_true",
                        help="Visualize only points without tour lines")
    parser.add_argument("--lines-only", action="store_true",
                        help="Visualize only tour lines without points")
    parser.add_argument("--show-points", action="store_true",
                        help="Show points on top of tour lines "
                             "(for tour mode)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Size of points when visualizing points only "
                             "(default: 1.0)")
    parser.add_argument("--line-width", type=float, default=2.0,
                        help="Width of tour lines (default: 2.0)")
    parser.add_argument("--line-color", default="black",
                        help="Color of tour lines (default: black)")

    args = parser.parse_args()

    # Determine TSP file path (support both positional and --tsp-path)
    tsp_path = args.tsp_path or args.tsp_path_alt
    if not tsp_path:
        # Look for TSP files in default location if none specified
        default_patterns = [
            "stipplings/tsp/*.tsp",
            "*.tsp"
        ]
        import glob
        for pattern in default_patterns:
            matches = glob.glob(pattern)
            if matches:
                tsp_path = matches[0]  # Use first match
                print(f"Using default TSP file: {tsp_path}")
                break

        if not tsp_path:
            parser.error("TSP file path is required. Specify as positional "
                         "argument or use --tsp-path")

    try:
        # Load coordinates
        coords = load_tsp_coordinates(tsp_path)
        print(f"Loaded {len(coords)} coordinates from {tsp_path}")

        # Validate conflicting options
        mode_flags = [args.points_only, args.lines_only]
        if sum(mode_flags) > 1:
            parser.error("Cannot specify multiple visualization modes "
                         "(--points-only, --lines-only)")

        # Determine visualization mode
        if args.points_only or (not args.tour_path and not args.lines_only):
            # Points-only visualization
            plot_points_only(coords, args.output, args.point_size)
        elif args.lines_only or args.tour_path:
            # Tour visualization (lines only or lines with optional points)
            if not args.tour_path:
                parser.error("--lines-only requires --tour-path to be "
                             "specified")

            tour = load_linkern_tour(args.tour_path)
            print(f"Loaded tour with {len(tour)} nodes from {args.tour_path}")

            # Lines-only mode: never show points unless explicitly requested
            show_points = args.show_points and not args.lines_only

            plot_tour(coords, tour, args.output, args.line_width,
                      args.line_color, show_points, args.point_size)
        else:
            # Default case: show points only if no tour provided
            plot_points_only(coords, args.output, args.point_size)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
