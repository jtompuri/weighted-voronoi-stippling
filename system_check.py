#!/usr/bin/env python3
"""
Check system capabilities for stippling optimization.
"""
import sys
import os
import platform
import subprocess


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version >= (3, 7):
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.7+ required for Numba optimization")
        return False


def check_cpu_info():
    """Check CPU information."""
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU count: {os.cpu_count()}")
    return True


def check_numba():
    """Check if Numba is available and working."""
    try:
        import numba
        print(f"✓ Numba version: {numba.__version__}")

        # Test JIT compilation
        @numba.jit(nopython=True)
        def test_func(x):
            return x * 2

        result = test_func(5)
        if result == 10:
            print("✓ Numba JIT compilation working")
            return True
        else:
            print("✗ Numba JIT compilation failed")
            return False
    except ImportError:
        print("✗ Numba not installed")
        print("  Install with: pip install numba")
        return False
    except Exception as e:
        print(f"✗ Numba error: {e}")
        return False


def check_gpu_support():
    """Check GPU availability and CUDA support."""
    gpu_available = False

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            # Parse GPU info
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'MiB' in line and '/' in line:
                    memory_info = f"{line.split()[-3]} / {line.split()[-1]}"
                    print(f"  GPU Memory: {memory_info}")
                    break
            gpu_available = True
        else:
            print("✗ NVIDIA GPU not detected or nvidia-smi not available")
    except FileNotFoundError:
        print("✗ nvidia-smi not found")

    # Check for CuPy
    try:
        import cupy as cp  # noqa: F401
        print(f"✓ CuPy version: {cp.__version__}")

        # Test basic GPU operation
        try:
            x = cp.array([1, 2, 3])
            y = cp.sum(x)
            if y == 6:
                print("✓ CuPy GPU operations working")
                gpu_available = True
            else:
                print("✗ CuPy GPU operations failed")
        except Exception as e:
            print(f"✗ CuPy GPU test failed: {e}")
    except ImportError:
        print("✗ CuPy not installed")
        if gpu_available:
            install_cmd = "pip install cupy-cuda11x  # or cupy-cuda12x"
            print(f"  Install with: {install_cmd}")

    return gpu_available


def check_dependencies():
    """Check required dependencies."""
    required = ['numpy', 'scipy', 'PIL', 'matplotlib', 'svgwrite']
    all_available = True

    for package in required:
        try:
            if package == 'PIL':
                import PIL
                print(f"✓ Pillow version: {PIL.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package} version: {version}")
        except ImportError:
            print(f"✗ {package} not installed")
            all_available = False

    return all_available


def estimate_performance(image_size, n_stipples, n_iter):
    """Estimate performance for different configurations."""
    pixels = image_size[0] * image_size[1]
    complexity = pixels * n_stipples * n_iter

    print(
        f"\nPerformance Estimates for {image_size[0]}x{image_size[1]} image:")
    print(f"Stipples: {n_stipples}, Iterations: {n_iter}")
    print(f"Computational complexity: {complexity:,.0f}")

    # Rough estimates based on benchmarks
    base_time = complexity / 1e8  # seconds

    print(f"Original implementation: ~{base_time:.1f}s")
    print(f"Optimized (Numba): ~{base_time * 0.2:.1f}s")
    print(f"Optimized (GPU): ~{base_time * 0.05:.1f}s")


def main():
    print("Stippling Performance Optimization - System Check")
    print("=" * 60)

    # System checks
    python_ok = check_python_version()
    check_cpu_info()  # Display CPU info (always succeeds)
    deps_ok = check_dependencies()
    numba_ok = check_numba()
    gpu_ok = check_gpu_support()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if python_ok and deps_ok:
        print("✓ Basic stippling functionality available")

        if numba_ok:
            print("✓ Use optimized-stippling.py for best CPU performance")
            if gpu_ok:
                print("✓ GPU acceleration available - expect 10-50x speedup")
            else:
                print("• Consider GPU setup for maximum performance")
                print("  - Install CUDA toolkit")
                print("  - Install CuPy: pip install cupy-cuda11x")
        else:
            print("• Install Numba for 5-15x speedup: pip install numba")
    else:
        print("✗ Install missing dependencies first")
        print("  pip install -r requirements.txt")

    # Performance estimates
    print("\n" + "=" * 60)
    estimate_performance((1024, 1024), 10000, 30)


if __name__ == "__main__":
    main()
