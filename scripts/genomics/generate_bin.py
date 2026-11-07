import numpy as np
import pandas as pd

def generate_and_save_matrix(rows, cols, filename):
    # Generate a matrix of random floats using numpy
    matrix = np.random.rand(rows, cols).astype(np.float32)
    
    # Convert the numpy matrix to a pandas DataFrame
    df = pd.DataFrame(matrix)
    
    # Save the DataFrame to a binary file
    with open(filename, 'wb') as f:
        # Convert the DataFrame to a numpy array and write it as binary
        f.write(df.values.tobytes())

# Example usage:
rows, cols = 10, 10  # Size of the matrix
filename = 'genotype1.bin'  # Output binary file name
generate_and_save_matrix(rows, cols, filename)

print(f"Random float matrix of size {rows}x{cols} has been written to {filename}")
