#!/usr/bin/env python3
"""
CUDA GEMM Benchmark to Markdown Tables Converter

This script parses the output from ./build/bin/cuda_benchmark and generates two Markdown tables:
1. TFLOPS performance table
2. Speedup performance table

Usage:
    ./build/bin/cuda_benchmark | python scripts/benchmark_to_md.py
    python scripts/benchmark_to_md.py < output.txt
    cat output.txt | python scripts/benchmark_to_md.py
"""

import sys
import re
from collections import defaultdict, OrderedDict


def parse_benchmark_output(lines):
    """
    Parse the benchmark output and extract function performance data.
    
    Returns:
        dict: {problem_size: {function_name: {'tflops': float, 'speedup': float}}}
    """
    data = OrderedDict()
    current_problem_size = None
    functions = set()
    
    for line in lines:
        line = line.strip()
        
        # Match problem size line: "Problem size: M=512, N=512, K=512"
        problem_match = re.match(r'Problem size: M=(\d+), N=(\d+), K=(\d+)', line)
        if problem_match:
            m, n, k = problem_match.groups()
            current_problem_size = f"M={m}, N={n}, K={k}"
            if current_problem_size not in data:
                data[current_problem_size] = {}
            continue
        
        # Skip header and separator lines
        if (line.startswith('Function') or 
            line.startswith('=') or 
            line.startswith('-') or
            not line or
            'CUDA SM80' in line or
            'Testing precisions' in line or
            'GPU:' in line or
            'Compute Capability' in line or
            'Precision:' in line or
            'Warmup runs:' in line or
            'Benchmark runs:' in line):
            continue
        
        # Parse function performance line
        # Format: function_name  time(ms)  tflops  bandwidth  correctness  max_error  speedup
        parts = line.split()
        if len(parts) >= 7 and current_problem_size:
            try:
                function_name = parts[0]
                time_ms = float(parts[1])
                tflops = float(parts[2])
                # Skip bandwidth (parts[3])
                correctness = parts[4]
                # Skip max_error (parts[5])
                speedup = float(parts[6])
                
                functions.add(function_name)
                data[current_problem_size][function_name] = {
                    'tflops': tflops,
                    'speedup': speedup,
                    'time_ms': time_ms,
                    'correctness': correctness
                }
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
    
    return data, sorted(functions)


def generate_markdown_table(data, functions, metric):
    """
    Generate a Markdown table for the specified metric.
    
    Args:
        data: Parsed benchmark data
        functions: List of function names
        metric: 'tflops' or 'speedup'
    
    Returns:
        str: Markdown table
    """
    if not data or not functions:
        return f"No data available for {metric.upper()} table.\n"
    
    # Table header
    header = "| Problem Size | " + " | ".join(functions) + " |\n"
    separator = "|" + "|".join([" --- "] * (len(functions) + 1)) + "|\n"
    
    table = header + separator
    
    # Table rows
    for problem_size, problem_data in data.items():
        row = f"| {problem_size} |"
        for function in functions:
            if function in problem_data:
                value = problem_data[function][metric]
                if metric == 'tflops':
                    row += f" {value:.3f} |"
                elif metric == 'speedup':
                    row += f" {value:.2f}x |"
                else:
                    row += f" {value} |"
            else:
                row += " N/A |"
        table += row + "\n"
    
    return table


def main():
    """Main function to process input and generate Markdown tables."""
    
    # Read input from stdin
    try:
        if sys.stdin.isatty():
            print("Error: No input provided. Please pipe the benchmark output to this script.", file=sys.stderr)
            print("Usage: ./build/bin/cuda_benchmark | python scripts/benchmark_to_md.py", file=sys.stderr)
            sys.exit(1)
        
        lines = sys.stdin.readlines()
    except KeyboardInterrupt:
        sys.exit(1)
    
    if not lines:
        print("Error: No input data received.", file=sys.stderr)
        sys.exit(1)
    
    # Parse the benchmark output
    data, functions = parse_benchmark_output(lines)
    
    if not data:
        print("Error: No valid benchmark data found in input.", file=sys.stderr)
        sys.exit(1)
    
    # Generate and print the tables
    print("# CUDA GEMM Benchmark Results\n")
    
    print("## TFLOPS Performance\n")
    tflops_table = generate_markdown_table(data, functions, 'tflops')
    print(tflops_table)
    
    print("## Speedup Performance\n")
    speedup_table = generate_markdown_table(data, functions, 'speedup')
    print(speedup_table)
    
    # Optional: Add a summary section
    print("## Summary\n")
    print(f"- **Total Problem Sizes Tested**: {len(data)}")
    print(f"- **Functions Benchmarked**: {len(functions)}")
    print(f"- **Functions**: {', '.join(functions)}")
    
    # Find best performing function for each metric
    best_tflops = defaultdict(list)
    best_speedup = defaultdict(list)
    
    for problem_size, problem_data in data.items():
        max_tflops = max(problem_data.values(), key=lambda x: x['tflops'])['tflops']
        max_speedup = max(problem_data.values(), key=lambda x: x['speedup'])['speedup']
        
        for func, metrics in problem_data.items():
            if metrics['tflops'] == max_tflops:
                best_tflops[func].append(problem_size)
            if metrics['speedup'] == max_speedup:
                best_speedup[func].append(problem_size)
    
    if best_tflops:
        print(f"\n### Best TFLOPS Performance")
        for func, sizes in best_tflops.items():
            print(f"- **{func}**: {len(sizes)} problem size(s)")
    
    if best_speedup:
        print(f"\n### Best Speedup Performance")
        for func, sizes in best_speedup.items():
            print(f"- **{func}**: {len(sizes)} problem size(s)")


if __name__ == "__main__":
    main()
