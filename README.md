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

**Tour lines (supports TSPLIB TOUR, Concorde .sol, and linkern output):**
```bash
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path path/to/photo.tour
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path photo_5000.sol
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path photo_5000.opt.tour
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path photo_5000.heu.tour
```

Accepted tour formats:
- TSPLIB TOUR files (`.tour`, `.opt.tour`, `.heu.tour`)
- Concorde integer-sequence outputs (`.sol` and similar)
- Concorde linkern tour output

`--tour-format` defaults to `auto`, so you usually only need `--tour-path`.
Use `--tour-format` and `--tour-index-base` only for troubleshooting format/base detection.

### TSP Solver Workflows

To create continuous line drawings, solve the TSP using external solvers:

**linkern (fast heuristic):**
```bash
linkern -o visualizations/tour/photo_5000.heu.tour stipplings/tsp/photo_5000.tsp
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path visualizations/tour/photo_5000.heu.tour --output visualizations/png/photo_5000_linkern.png
```

**Concorde (default `.sol` output):**
```bash
concorde stipplings/tsp/photo_5000.tsp
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path photo_5000.sol --output visualizations/png/photo_5000_concorde.png
```

**Concorde (explicit TSPLIB TOUR output with `-o`):**
```bash
concorde -o visualizations/tour/photo_5000.opt.tour stipplings/tsp/photo_5000.tsp
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path visualizations/tour/photo_5000.opt.tour --output visualizations/png/photo_5000_concorde.png
```

**LKH (parameter-file based):**
- LKH usually runs from a parameter file instead of direct CLI flags.
- Include `PROBLEM_FILE = stipplings/tsp/photo_5000.tsp`.
- Include a tour-output setting such as `TOUR_FILE = ...` or `OUTPUT_TOUR_FILE = ...` depending on your LKH version/configuration.
- Then visualize the produced tour file:
```bash
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path path/to/lkh-output.tour
```

**Custom LK solver output:**
- Custom LK outputs `.heu.tour` in TSPLIB TOUR format.
```bash
python visualize.py stipplings/tsp/photo_5000.tsp --tour-path path/to/photo_5000.heu.tour --output visualizations/png/photo_5000_custom_lk.png
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
| `--tour-path` | Path to TSPLIB TOUR/.sol/linkern tour file |
| `--tour-format` | `auto`, `tsplib`, `concorde-sol`, `linkern` |
| `--tour-index-base` | Concorde index base: `auto`, `0`, `1` |
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

- [Secord, A. (2002). "Weighted Voronoi stippling"](https://www.cs.ubc.ca/labs/imager/tr/2002/secord2002b/secord.2002b.pdf)
- [TSP Art by Robert Bosch](https://www2.oberlin.edu/math/faculty/bosch/tspart-page.html)
- [Numba Documentation](https://numba.pydata.org/)
