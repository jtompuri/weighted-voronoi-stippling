# Weighted Voronoi Stippling

High-performance implementation of the Weighted Voronoi Stippling algorithm for converting images into artistic stipple representations. Uses Numba JIT compilation for 5-15x speedup on multi-core CPUs.

## Example Output

| Original | Stipples | Tour Visualization |
|----------|----------|-------------------|
| ![Original](/images/example-1024px.png) | ![Stipples](/images/example-1024px_10000.png) | ![Tour](/images/example-1024px_10000v.png) |

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate stipples:**
   ```bash
   python stippling.py images/example-1024px.png --stipples 10000
   ```

3. **Visualize points:**
   ```bash
   python visualize.py stipplings/tsp/example-1024px_10000.tsp --points-only
   ```

## Usage

### Basic Stippling

Generate stipples from any image:
```bash
# Default settings (5000 stipples, 30 iterations)
python stippling.py images/photo.jpg

# Custom settings
python stippling.py images/photo.jpg --stipples 10000 --iter 50 --radius 2.0
```

**Output files:**
- `stipplings/png/photo_5000.png` - Visual stipple image
- `stipplings/tsp/photo_5000.tsp` - Coordinate data for TSP solvers

### Visualization

**Points only:**
```bash
python visualize.py stipplings/tsp/photo_5000.tsp --points-only
```

**Tour lines (requires .tour file from TSP solver):**
```bash
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path path/to/photo.tour
```

### TSP Integration for Line Art

To create continuous line drawings, solve the TSP using external solvers:

**Option 1: Lin-Kernighan (fast approximation)**
```bash
linkern stipplings/tsp/photo_5000.tsp
mv photo_5000.tour visualizations/tour/
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path visualizations/tour/photo_5000.tour
```

**Option 2: Concorde (optimal solution)**
```bash
concorde stipplings/tsp/photo_5000.tsp
# Convert .sol to .tour format if needed
```

**Option 3: Online solvers**
Upload your `.tsp` file to [NEOS Server](https://neos-server.org/neos/solvers/co:concorde/TSP.html) for optimal solutions.

## Command Reference

### stippling.py
| Option | Default | Description |
|--------|---------|-------------|
| `--stipples` | 5000 | Number of stipples (1-1,000,000) |
| `--iter` | 30 | Lloyd relaxation iterations (1-1000) |
| `--radius` | 1.0 | Stipple radius in pixels (0.1-100) |
| `--no-numba` | - | Disable JIT compilation |
| `--verbose` | - | Detailed progress logging |

### visualize.py
| Option | Description |
|--------|-------------|
| `--points-only` | Show only stipple points |
| `--lines-only` | Show only tour lines |
| `--tour-path` | Path to .tour file |
| `--output` | Save to file instead of display |
| `--point-size` | Point size (default: 1.0) |
| `--line-width` | Line width (default: 2.0) |

## File Organization

```
weighted-voronoi-stippling/
├── images/              # Input images (tracked by git)
├── stipplings/
│   ├── png/            # Generated stipple images
│   └── tsp/            # TSP coordinate files
├── visualizations/
│   ├── tour/           # TSP solution files (.tour)
│   └── png/            # Tour visualization images
├── stippling.py         # Main stippling algorithm
└── visualize.py         # Visualization tool
```

Files are automatically named with stipple counts: `example_5000.png`, `example_5000.tsp`, etc.

## Technical Details

- **Algorithm**: Weighted Voronoi stippling with Lloyd relaxation
- **Performance**: 5-15x speedup with Numba JIT compilation
- **TSP Format**: Standard TSPLIB format for compatibility with solvers
- **Fallback**: Pure Python mode when Numba unavailable

## References

- Secord, A. (2002). "Weighted Voronoi stippling"
- [TSP Art by Robert Bosch](https://www2.oberlin.edu/math/faculty/bosch/tspart-page.html)
- [Numba Documentation](https://numba.pydata.org/)