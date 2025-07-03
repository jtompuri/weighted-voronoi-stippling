# Weighted Voronoi Stippling

This repository contains a high-performance CPU implementation of the Weighted Voronoi Stippling algorithm (Secord 2002) for converting grayscale images into artistic stipple representations.

## Example output

**Original example photo (1024 x 1024 px):**

![Original photo of a boy](/images/example-1024px.png)

**Voronoi stippling of the photo with 10 000 stipples:**

![Voronoi stippling of the photo](/stipplings/png/example-1024px_10000.png)

**Visualization of the tour created by an external Lin-Kernihan TSP Solver:**

![Visualization of the tour](/visualizations/png/example-1024px_10000.png)

## Features

The implementation uses Numba JIT compilation to achieve excellent performance on CPU:

### 1. Numba JIT Compilation
- **Rejection Sampling**: ~10-50x speedup using `@jit(nopython=True, parallel=True)`
- **Voronoi Computation**: Custom implementation with parallel loops for optimal CPU utilization
- **Centroid Computation**: Batch processing all centroids in parallel
- **Pure Python Fallback**: Automatic fallback to pure Python if Numba is unavailable

### 2. Algorithmic Improvements
- **Batch Processing**: Compute all centroids simultaneously instead of one-by-one
- **Memory Optimization**: Avoid creating individual masks for each point
- **Efficient Data Structures**: Use contiguous arrays for better cache performance
- **Parallel Processing**: Leverage all CPU cores through Numba's parallel execution

### 3. Output Formats
- **PNG Images**: Visual representation of stipples as black dots on white background
- **TSP Files**: Standard TSPLIB format for traveling salesman optimization

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

The project requires:
- **NumPy**: For numerical computations
- **Numba**: For JIT compilation and parallel processing
- **Pillow (PIL)**: For image I/O operations

All dependencies are automatically installed from `requirements.txt`.

## Usage

### Basic Stippling

```bash
# Basic usage with default settings
python stippling.py images/example-512px.png

# Specify number of stipples and iterations
python stippling.py images/example-512px.png --stipples 5000 --iter 30

# Control stipple size and output location
python stippling.py images/example-512px.png --stipples 2000 --radius 2.0 --iter 20
```

### Performance Options

```bash
# Disable Numba JIT compilation (use pure Python fallback)
python stippling.py images/example-512px.png --no-numba

# Enable verbose logging
python stippling.py images/example-512px.png --verbose
```

### Command Line Options

The main `stippling.py` script supports the following options:

- `image`: Path to input grayscale image (required)
- `--output`: Base name for output files (default: derived from input filename)
- `--stipples`: Number of stipples to generate (default: 5000)
- `--radius`: Radius of each stipple in pixels (default: 1.0)
- `--iter`: Number of Lloyd relaxation iterations (default: 30)
- `--no-numba`: Disable Numba JIT compilation (use pure Python fallback)
- `--verbose`: Enable detailed logging

Run `python stippling.py --help` for complete usage information.

## Key Optimizations Explained

### 1. Numba JIT Compilation
```python
@jit(nopython=True, parallel=True)
def compute_voronoi_labels_numba(points, h, w):
    # Compiled to native code, runs in parallel across CPU cores
    # Achieves 5-10x speedup over pure Python
```

### 2. Parallel Voronoi Computation
```python
# Parallel distance computation across all pixels
# Each CPU core processes a subset of image pixels
labels = np.empty((h, w), dtype=np.int32)
for i in prange(h):  # Parallel loop
    for j in prange(w):
        # Find closest stipple point
```

### 3. Batch Centroid Computation
```python
# Instead of: for each point, compute centroid
# Do: compute all centroids simultaneously in parallel
centroids = compute_centroids_batch_numba(labels, rho, n_points)
```

### 4. Automatic Fallback
The implementation automatically falls back to pure Python if Numba is not available, ensuring compatibility across all systems while maintaining optimal performance when possible.

## References

- [TSP Art by Robert Bosch](https://www2.oberlin.edu/math/faculty/bosch/tspart-page.html)
- Secord, A. (2002). "Weighted Voronoi stippling"
- Lloyd, S. (1982). "Least squares quantization in PCM"
- Numba Documentation: https://numba.pydata.org/

## TSP (Traveling Salesman Problem) Integration

The stippling algorithm automatically generates TSP files containing the stipple coordinates. This enables you to:

1. **Optimize stipple drawing order** using TSP solvers
2. **Minimize pen travel distance** for physical plotting
3. **Create efficient drawing paths** for CNC machines or plotters

### TSP File Format

The generated `.tsp` files follow the standard TSPLIB format:
```
NAME: INPUT_FILENAME
TYPE: TSP
COMMENT: Stipple points for Traveling Salesman Problem
DIMENSION: 5000
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 245.123456 167.891234
2 312.456789 298.765432
...
EOF
```

### Using TSP Files

**Analyze a TSP instance:**
```bash
python tsp_utils.py stipplings/tsp/stipples.tsp --analyze
```

**Visualize stipple points:**
```bash
python tsp_utils.py stipplings/tsp/stipples.tsp --visualize --output tour_viz.png
```

**Generate a simple tour with nearest neighbor:**
```bash
python tsp_utils.py stipplings/tsp/stipples.tsp --nearest-neighbor --visualize
```

**Export for professional TSP solvers (e.g., Concorde):**
```bash
python tsp_utils.py stipplings/tsp/stipples.tsp --export-concorde
```

### Professional TSP Solvers

For optimal results with large stipple sets, consider:

1. **Concorde TSP Solver**: Industry-standard exact TSP solver
   ```bash
   # After exporting with tsp_utils.py
   concorde example-512px_5000_concorde.tsp
   ```

2. **Lin-Kernighan Solver within Concorde package**: Fast heuristic for large instances
   ```bash
   linkern example-512px_5000_concorde.tsp
   ```

## Workflow and File Organization

### Folder Structure

The project uses an organized folder structure to clearly separate different types of outputs:

```
weighted-voronoi-stippling/
├── images/              # Input images (source photos)
├── stipplings/          # Stippling algorithm outputs
│   ├── png/            # Generated stipple images (.png)
│   └── tsp/            # TSP problem files (.tsp)
├── visualizations/      # Tour visualization outputs
│   ├── tour/           # TSP solution files (.tour)
│   └── png/            # Tour visualization images (.png)
├── stippling.py         # Main stippling algorithm
├── visualize.py         # Tour visualization
└── tsp_utils.py         # TSP analysis utilities
```

### Complete Workflow

1. **Generate stipples and TSP file:**
   ```bash
   python stippling.py images/example-512px.png --stipples 5000 --iter 30
   # Creates: stipplings/png/example-512px_5000.png
   #          stipplings/tsp/example-512px_5000.tsp
   ```

2. **Solve TSP with external solver (e.g., Lin-Kernighan):**
   ```bash
   # In your TSP solver repository:
   linkern stipplings/tsp/example-512px_5000.tsp
   # Creates: example-512px_5000.tour
   ```

3. **Move tour file to visualizations folder:**
   ```bash
   mv example-512px_5000.tour visualizations/tour/
   ```

4. **Visualize the optimized tour:**
   ```bash
   python visualize.py --tsp-path stipplings/tsp/example-512px_5000.tsp \
                       --tour-path visualizations/tour/example-512px_5000.tour \
                       --output visualizations/png/example_tour_5000.png
   ```

### File Naming Convention

Files follow a consistent naming pattern that includes the stipple count:
- Input: `images/example-512px.png`
- Stipples: `stipplings/png/example-512px_5000.png` (includes stipple count)
- TSP: `stipplings/tsp/example-512px_5000.tsp` (includes stipple count)
- Tour: `visualizations/tour/example-512px_5000.tour` (matches TSP file)
- Visualization: `visualizations/png/example_tour_visualization.png`

The stipple count is automatically included in the output filenames to help distinguish between different stipple densities of the same input image.
