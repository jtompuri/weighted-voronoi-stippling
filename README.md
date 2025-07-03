# Weighted Voronoi Stippling

This repository contains a high-performance implementation of the Weighted Voronoi Stippling algorithm (Secord 2002) with GPU and CPU optimizations for converting grayscale images into artistic stipple representations.

## Example output

[Original photo of a boy](/images/example-1024px.png)

[Voronoi stippling of the photo](/stipplings/png/stipples_example-1024px_10000.png)

[Visualization of the tour](/visualizations/png/example-1024px_10000.png)

## Features

The implementation combines multiple optimization strategies in a single, unified codebase:

### 1. Numba JIT Compilation
- **Rejection Sampling**: ~10-50x speedup using `@jit(nopython=True, parallel=True)`
- **Voronoi Computation**: Custom implementation with parallel loops
- **Centroid Computation**: Batch processing all centroids in parallel

### 2. GPU Acceleration (Optional)
- **CuPy Integration**: GPU-accelerated Voronoi diagram computation using CUDA
- **Vectorized Operations**: Leverage GPU's parallel processing for distance calculations
- **Memory Management**: Efficient GPU memory allocation and data transfer

### 3. Algorithmic Improvements
- **Batch Processing**: Compute all centroids simultaneously instead of one-by-one
- **Memory Optimization**: Avoid creating individual masks for each point
- **Efficient Data Structures**: Use contiguous arrays for better cache performance

### 4. Output Formats
- **PNG Images**: Visual representation of stipples as black dots on white background
- **TSP Files**: Standard TSPLIB format for traveling salesman optimization

## Installation

1. Install basic dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU acceleration (optional):
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x  
pip install cupy-cuda12x
```

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
# Disable GPU acceleration (use CPU only)
python stippling.py images/example-512px.png --no-gpu

# Disable Numba JIT compilation
python stippling.py images/example-512px.png --no-numba

# Disable all optimizations for debugging
python stippling.py images/example-512px.png --no-gpu --no-numba

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
- `--no-gpu`: Disable GPU acceleration
- `--no-numba`: Disable Numba JIT compilation
- `--verbose`: Enable detailed logging

Run `python stippling.py --help` for complete usage information.

## Key Optimizations Explained

### 1. Numba JIT Compilation
```python
@jit(nopython=True, parallel=True)
def compute_voronoi_labels_numba(points, h, w):
    # Compiled to native code, runs in parallel
```

### 2. GPU Vectorization
```python
# CPU: O(n*pixels) serial computation
# GPU: O(pixels) parallel computation with broadcasting
distances = cp.sum((coords_expanded - points_expanded)**2, axis=2)
```

### 3. Batch Centroid Computation
```python
# Instead of: for each point, compute centroid
# Do: compute all centroids simultaneously in parallel
centroids = compute_centroids_batch_numba(labels, rho, n_points)
```

## References

- Secord, A. (2002). "Weighted Voronoi stippling"
- Lloyd, S. (1982). "Least squares quantization in PCM"
- Numba Documentation: https://numba.pydata.org/
- CuPy Documentation: https://cupy.dev/

## TSP (Traveling Salesman Problem) Integration

The stippling algorithm automatically generates TSP files containing the stipple coordinates. This enables you to:

1. **Optimize stipple drawing order** using TSP solvers
2. **Minimize pen travel distance** for physical plotting
3. **Create efficient drawing paths** for CNC machines or plotters

### TSP File Format

The generated `.tsp` files follow the standard TSPLIB format:
```
NAME: STIPPLES_NAME
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
   concorde stipples_concorde.tsp
   ```

2. **Lin-Kernighan Solver within Concorde package**: Fast heuristic for large instances
   ```bash
   linkern stipples_concorde.tsp
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
   # Creates: stipplings/png/stipples_example-512px_5000.png
   #          stipplings/tsp/stipples_example-512px_5000.tsp
   ```

2. **Solve TSP with external solver (e.g., Lin-Kernighan):**
   ```bash
   # In your TSP solver repository:
   linkern stipplings/tsp/stipples_example-512px_5000.tsp
   # Creates: stipples_example-512px_5000.tour
   ```

3. **Move tour file to visualizations folder:**
   ```bash
   mv stipples_example-512px_5000.tour visualizations/tour/
   ```

4. **Visualize the optimized tour:**
   ```bash
   python visualize.py --tsp-path stipplings/tsp/stipples_example-512px_5000.tsp \
                       --tour-path visualizations/tour/stipples_example-512px_5000.tour \
                       --output visualizations/png/example_tour_5000.png
   ```

### File Naming Convention

Files follow a consistent naming pattern that includes the stipple count:
- Input: `images/example-512px.png`
- Stipples: `stipplings/png/stipples_example-512px_5000.png` (includes stipple count)
- TSP: `stipplings/tsp/stipples_example-512px_5000.tsp` (includes stipple count)
- Tour: `visualizations/tour/stipples_example-512px_5000.tour` (matches TSP file)
- Visualization: `visualizations/png/example_tour_visualization.png`

The stipple count is automatically included in the output filenames to help distinguish between different stipple densities of the same input image.
