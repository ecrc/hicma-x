#!/usr/bin/env python3
"""
Memory computation script for HiCMA PaRSEC operations.

This script calculates memory requirements for matrix operations including:
- Dense matrix memory (Ag)
- Low-rank matrix memory (uv)
- Memory per node or number of nodes needed

@copyright (c) 2023-2025     Saint Louis University (SLU)
@copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
@copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
@copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
                              All rights reserved.
"""

import argparse
import sys

# Memory constant (128 GB)
MEM = 128

def calculate_memory(matrix_size, tile_size, maxrank, nodes=None):
    """
    Calculate memory requirements for matrix operations.
    
    Args:
        matrix_size (int): Dimension (N) of the matrices
        tile_size (int): Dimension (NB) of the tiles
        maxrank (int): Maximum rank of the tiles
        nodes (int, optional): Number of nodes
    
    Returns:
        tuple: (Ag, uv, memory_per_node_or_nodes_needed)
    """
    # Convert memory to bytes
    memory = MEM * 1024**3
    
    # Calculate number of tiles
    NT = (matrix_size + tile_size - 1) // tile_size  # Ceiling division
    
    # Calculate dense matrix memory (Ag) in bytes
    Ag = NT * tile_size * tile_size * 8  # 8 bytes per double
    
    # Calculate low-rank matrix memory (uv) in bytes
    uv = (NT * (NT - 1)) * tile_size * maxrank * 8  # 8 bytes per double
    
    if nodes:
        # Calculate memory per node
        memory_nodes = nodes * 1024**3
        memory_per_node = (Ag + uv) / memory_nodes
        return Ag, uv, memory_per_node
    else:
        # Calculate number of nodes needed
        nodes_needed = (Ag + uv) / memory
        return Ag, uv, nodes_needed

def main():
    """Main function to parse arguments and calculate memory requirements."""
    parser = argparse.ArgumentParser(
        description="Calculate memory requirements for HiCMA PaRSEC matrix operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -N 1024 -t 64 -u 32 -a 4    # Calculate memory per node for 4 nodes
  %(prog)s -N 1024 -t 64 -u 32         # Calculate number of nodes needed
        """
    )
    
    parser.add_argument('-N', '--matrix-size', type=int, default=16,
                       help='Dimension (N) of the matrices (default: 16)')
    parser.add_argument('-t', '--tile-size', type=int, default=4,
                       help='Dimension (NB) of the tiles (default: 4)')
    parser.add_argument('-u', '--maxrank', type=int, default=4,
                       help='Maximum rank of the tiles (default: 4)')
    parser.add_argument('-a', '--nodes', type=int, default=None,
                       help='Number of nodes (if specified, calculates memory per node)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.matrix_size <= 0:
        print("Error: Matrix size must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.tile_size <= 0:
        print("Error: Tile size must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.maxrank <= 0:
        print("Error: Maxrank must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.nodes is not None and args.nodes <= 0:
        print("Error: Number of nodes must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Calculate memory requirements
    try:
        Ag, uv, result = calculate_memory(
            args.matrix_size, 
            args.tile_size, 
            args.maxrank, 
            args.nodes
        )
        
        # Convert to GB for display
        Ag_gb = Ag / (1024**3)
        uv_gb = uv / (1024**3)
        
        if args.nodes:
            # Output memory per node
            print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}, "
                  f"No. of {args.nodes} nodes, if maxrank is {args.maxrank} "
                  f"each node requires {result:.6f} GB")
        else:
            # Output number of nodes needed
            print(f"Matrix size {args.matrix_size} of tile size {args.tile_size}: "
                  f"if maxrank is {args.maxrank} it needs No. of {result:.6f} nodes")
        
        # Additional detailed information
        print(f"\nDetailed breakdown:")
        print(f"  Dense matrix memory (Ag): {Ag_gb:.6f} GB")
        print(f"  Low-rank matrix memory (uv): {uv_gb:.6f} GB")
        print(f"  Total memory: {Ag_gb + uv_gb:.6f} GB")
        
    except Exception as e:
        print(f"Error calculating memory requirements: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
