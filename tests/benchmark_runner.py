#!/usr/bin/env python
"""
Benchmark runner for comparing different ProteinMPNN configurations.

Usage:
    python benchmark_runner.py --baseline
    python benchmark_runner.py --compare baseline_results.json
    python benchmark_runner.py --quick  # Quick benchmark with small workloads
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
from protein_mpnn import ProteinMPNN


# Get test PDB path
def get_test_pdb_path():
    """Get path to test PDB file (6MRR - 68 residue protein)."""
    # Try relative to script location
    script_dir = Path(__file__).parent
    pdb_path = script_dir.parent / "inputs" / "PDB_monomers" / "pdbs" / "6MRR.pdb"

    if not pdb_path.exists():
        raise FileNotFoundError(
            f"Test PDB file not found at {pdb_path}. "
            "Make sure inputs/PDB_monomers/pdbs/6MRR.pdb exists in the repo."
        )

    return str(pdb_path)


class BenchmarkConfig:
    """Configuration for a benchmark run."""

    def __init__(self, name: str, num_structures: int, num_seq_per_target: int,
                 batch_size: int, temperatures: str = "0.1"):
        self.name = name
        self.num_structures = num_structures
        self.num_seq_per_target = num_seq_per_target
        self.batch_size = batch_size
        self.temperatures = temperatures

    def to_dict(self):
        return {
            'name': self.name,
            'num_structures': self.num_structures,
            'num_seq_per_target': self.num_seq_per_target,
            'batch_size': self.batch_size,
            'temperatures': self.temperatures,
        }


# Standard benchmark configurations
QUICK_BENCHMARKS = [
    BenchmarkConfig("single_seq", 1, 1, 1),
    BenchmarkConfig("small_batch", 1, 5, 1),
    BenchmarkConfig("multi_struct_small", 3, 1, 1),
]

STANDARD_BENCHMARKS = [
    BenchmarkConfig("single_seq", 1, 1, 1),
    BenchmarkConfig("medium_batch", 1, 10, 1),
    BenchmarkConfig("large_batch", 1, 20, 1),
    BenchmarkConfig("batch_size_5", 1, 10, 5),
    BenchmarkConfig("multi_struct_5x1", 5, 1, 1),
    BenchmarkConfig("multi_struct_5x10", 5, 10, 1),
    BenchmarkConfig("multi_temp", 1, 3, 1, "0.1 0.2 0.5"),
]

COMPREHENSIVE_BENCHMARKS = STANDARD_BENCHMARKS + [
    BenchmarkConfig("large_batch_2", 1, 50, 1),
    BenchmarkConfig("batch_size_10", 1, 20, 10),
    BenchmarkConfig("multi_struct_10x10", 10, 10, 1),
    BenchmarkConfig("multi_temp_large", 1, 9, 1, "0.1 0.2 0.3"),
]


def run_benchmark(model: ProteinMPNN, config: BenchmarkConfig,
                  warmup: bool = True, n_runs: int = 3, test_pdb_path: str = None) -> Dict:
    """Run a single benchmark configuration."""
    if test_pdb_path is None:
        test_pdb_path = get_test_pdb_path()

    print(f"\n{'='*60}")
    print(f"Benchmark: {config.name}")
    print(f"Config: {config.num_structures} structures, "
          f"{config.num_seq_per_target} seq/struct, "
          f"batch_size={config.batch_size}")
    print('='*60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times = []

    # Warmup run
    if warmup:
        print("Warming up...")
        model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=1,
            batch_size=1,
            sampling_temp="0.1",
            seed=42
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Actual benchmark runs
    for run in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start_time = time.time()

        # Run the workload
        for struct_idx in range(config.num_structures):
            result = model.sample(
                pdb_path_or_str=test_pdb_path,
                num_seq_per_target=config.num_seq_per_target,
                batch_size=config.batch_size,
                sampling_temp=config.temperatures,
                seed=42 + struct_idx
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {run + 1}: {elapsed:.3f}s")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate throughput
    total_sequences = config.num_structures * config.num_seq_per_target
    num_temps = len(config.temperatures.split())
    total_sequences *= num_temps
    throughput = total_sequences / avg_time

    # GPU memory if available
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    results = {
        'config': config.to_dict(),
        'times': times,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': (max_time - min_time) / 2,  # Simple std approximation
        'total_sequences': total_sequences,
        'throughput': throughput,
        'time_per_sequence': avg_time / total_sequences,
        'peak_memory_mb': peak_memory_mb,
    }

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.3f}s ± {results['std_time']:.3f}s")
    print(f"  Time per sequence: {results['time_per_sequence']:.4f}s")
    print(f"  Throughput: {throughput:.2f} seq/s")
    if peak_memory_mb > 0:
        print(f"  Peak GPU memory: {peak_memory_mb:.1f} MB")

    return results


def save_results(results: List[Dict], output_file: Path):
    """Save benchmark results to JSON."""
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'benchmarks': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print('='*60)


def compare_results(baseline_file: Path, current_results: List[Dict]):
    """Compare current results with baseline."""
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    baseline_benchmarks = {b['config']['name']: b for b in baseline_data['benchmarks']}

    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINE")
    print('='*60)
    print(f"Baseline: {baseline_data['timestamp']} on {baseline_data['device']}")
    print(f"Current:  {time.strftime('%Y-%m-%d %H:%M:%S')} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print('='*60)

    print(f"\n{'Benchmark':<25} {'Baseline':<12} {'Current':<12} {'Speedup':<12} {'Status'}")
    print('-' * 70)

    for current in current_results:
        name = current['config']['name']
        if name in baseline_benchmarks:
            baseline = baseline_benchmarks[name]
            baseline_time = baseline['avg_time']
            current_time = current['avg_time']
            speedup = baseline_time / current_time
            change_pct = (1 - current_time / baseline_time) * 100

            # Determine status
            if speedup > 1.1:
                status = "✓ FASTER"
            elif speedup < 0.9:
                status = "✗ SLOWER"
            else:
                status = "≈ SAME"

            print(f"{name:<25} {baseline_time:>10.3f}s  {current_time:>10.3f}s  "
                  f"{speedup:>9.2f}x  {status}")
        else:
            print(f"{name:<25} {'N/A':<12} {current['avg_time']:>10.3f}s  {'N/A':<12} NEW")

    print('-' * 70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark ProteinMPNN performance')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark suite (faster)')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive benchmark suite (slower)')
    parser.add_argument('--baseline', action='store_true',
                        help='Save results as baseline for future comparisons')
    parser.add_argument('--compare', type=str, metavar='FILE',
                        help='Compare results with baseline file')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results (default: benchmark_results.json)')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='Run warmup iteration before benchmarking')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of runs per benchmark (default: 3)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto)')

    args = parser.parse_args()

    # Select benchmark suite
    if args.quick:
        benchmarks = QUICK_BENCHMARKS
        print("\nRunning QUICK benchmark suite...")
    elif args.comprehensive:
        benchmarks = COMPREHENSIVE_BENCHMARKS
        print("\nRunning COMPREHENSIVE benchmark suite...")
    else:
        benchmarks = STANDARD_BENCHMARKS
        print("\nRunning STANDARD benchmark suite...")

    # Initialize model
    print("\nInitializing ProteinMPNN model...")
    model = ProteinMPNN(
        model_name='v_48_020',
        device=args.device,
        suppress_print=True
    )
    print(f"Device: {model.device}")

    # Run benchmarks
    results = []
    for config in benchmarks:
        result = run_benchmark(model, config, warmup=args.warmup, n_runs=args.n_runs)
        results.append(result)

    # Save results
    output_file = Path(args.output)
    if args.baseline:
        test_dir = Path(__file__).parent
        output_file = test_dir / 'baseline_results.json'

    save_results(results, output_file)

    # Compare with baseline if requested
    if args.compare:
        compare_results(Path(args.compare), results)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    total_time = sum(r['avg_time'] for r in results)
    print(f"Total benchmark time: {total_time:.2f}s")
    print(f"Number of benchmarks: {len(results)}")
    print(f"Average per benchmark: {total_time/len(results):.2f}s")


if __name__ == '__main__':
    main()
