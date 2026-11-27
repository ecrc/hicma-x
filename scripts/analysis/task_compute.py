#!/usr/bin/env python3
"""
Task computation script for HiCMA PaRSEC operations.

This script calculates the number of tasks (operations) for each node in distributed
matrix computations including:
- POTRF operations (po)
- TRSM operations (trsm)
- SYRK operations (syrk)
- GEMM operations (gemm)

@copyright (c) 2023-2025     Saint Louis University (SLU)
@copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
@copyright (c) 2023-2025     King Abdullah University of Science and Technology (KAUST)
@copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
                              All rights reserved.
"""

import argparse
import sys

def calculate_tasks(matrix_size, tile_size, rank, nodes):
    """
    Calculate tasks for each node assuming load balanced distribution.
    
    Args:
        matrix_size (int): Dimension (N) of the matrices
        tile_size (int): Dimension (NB) of the tiles
        rank (int): Rank of the tiles
        nodes (int): Number of nodes
    
    Returns:
        tuple: (NT, po, trsm, syrk, gemm, total_count)
    """
    # Calculate number of tiles
    NT = (matrix_size + tile_size - 1) // tile_size  # Ceiling division
    
    # Calculate number of operations
    po = NT  # POTRF operations
    trsm = NT * (NT - 1) // 2  # TRSM operations
    syrk = NT * (NT - 1) // 2  # SYRK operations
    gemm = NT * (NT - 1) * (NT - 2) // 6  # GEMM operations
    
    # Calculate total operation count
    total_count = (po * tile_size**3 // 3 + 
                   trsm * tile_size**2 * rank + 
                   syrk * 2 * (tile_size**2 * rank + tile_size * rank**2 * 2))
    
    return NT, po, trsm, syrk, gemm, total_count

def main():
    """Main function to parse arguments and calculate task distribution."""
    parser = argparse.ArgumentParser(
        description="Calculate task distribution for HiCMA PaRSEC matrix operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -N 1024 -t 64 -u 32 -a 4         # Calculate tasks for 4 nodes
  %(prog)s -N 1024 -t 64 -u 32 -a 4 -v      # Verbose output with detailed breakdown
        """
    )
    
    parser.add_argument('-N', '--matrix-size', type=int, default=16,
                       help='Dimension (N) of the matrices (default: 16)')
    parser.add_argument('-t', '--tile-size', type=int, default=4,
                       help='Dimension (NB) of the tiles (default: 4)')
    parser.add_argument('-u', '--rank', type=int, default=2,
                       help='Rank of the tiles (default: 2)')
    parser.add_argument('-a', '--nodes', type=int, default=1,
                       help='Number of nodes (default: 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output with detailed breakdown')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.matrix_size <= 0:
        print("Error: Matrix size must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.tile_size <= 0:
        print("Error: Tile size must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.rank <= 0:
        print("Error: Rank must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.nodes <= 0:
        print("Error: Number of nodes must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Calculate tasks
    try:
        NT, po, trsm, syrk, gemm, total_count = calculate_tasks(
            args.matrix_size, 
            args.tile_size, 
            args.rank, 
            args.nodes
        )
        
        if args.verbose:
            # Detailed breakdown
            print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}, "
                  f"No. of {args.nodes} nodes, each node there are {po/args.nodes:.6f} po")
            print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}, "
                  f"No. of {args.nodes} nodes, each node there are {trsm/args.nodes:.6f} trsm")
            print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}, "
                  f"No. of {args.nodes} nodes, each node there are {syrk/args.nodes:.6f} syrk")
            print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}, "
                  f"No. of {args.nodes} nodes, each node there are {gemm/args.nodes:.6f} gemm")
        
        # Summary output
        print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}, "
              f"No. of {args.nodes} nodes, rank of {args.rank}: operations, "
              f"total: {total_count}, each node {total_count/args.nodes:.6g}")
        
        # Additional information
        if args.verbose:
            print(f"\nDetailed breakdown:")
            print(f"  Number of tiles (NT): {NT}")
            print(f"  POTRF operations: {po}")
            print(f"  TRSM operations: {trsm}")
            print(f"  SYRK operations: {syrk}")
            print(f"  GEMM operations: {gemm}")
            print(f"  Total operations: {total_count}")
            print(f"  Operations per node: {total_count/args.nodes:.6g}")
        
    except Exception as e:
        print(f"Error calculating task distribution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
