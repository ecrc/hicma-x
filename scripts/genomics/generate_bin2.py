import numpy as np
import pandas as pd

def generate_and_save_matrix(rows, cols, filename):
    # Generate a matrix of random floats using numpy
    matrix = np.random.randint(0, 3, size=(rows,cols)).astype(np.float32)
    print((matrix.dtype))
    total_sum = np.sum(matrix)
    print(total_sum)
    print(matrix)   
    # Convert the numpy matrix to a pandas DataFrame
    df = pd.DataFrame(matrix)
    
    # Save the DataFrame to a binary file
    with open(filename, 'wb') as f:
        # Convert the DataFrame to a numpy array and write it as binary
        f.write(df.values.tobytes())
    return matrix

# Example usage:
rows, cols = 1024, 2048  # Size of the matrix
filename = 'genotype1.bin'  # Output binary file name
mat1=generate_and_save_matrix(rows, cols, filename)
print(f"Random float matrix of size {rows}x{cols} has been written to {filename}")

# Example usage:
#rows, cols = 2048, 2048  # Size of the matrix
filename = 'genotype2.bin'  # Output binary file name
mat2=generate_and_save_matrix(rows, cols, filename)
print(f"Random float matrix of size {rows}x{cols} has been written to {filename}")

# Example usage:
#rows, cols = 2048, 2048   # Size of the matrix
filename = 'genotype3.bin'  # Output binary file name
mat3=generate_and_save_matrix(rows, cols, filename)
print(f"Random float matrix of size {rows}x{cols} has been written to {filename}")


# Example usage:
#rows, cols = 2048, 2048   # Size of the matrix
filename = 'genotype4.bin'  # Output binary file name
mat3=generate_and_save_matrix(rows, cols, filename)
print(f"Random float matrix of size {rows}x{cols} has been written to {filename}")


# Example usage:
#rows, cols = 2048, 2048   # Size of the matrix
filename = 'genotype5.bin'  # Output binary file name
mat3=generate_and_save_matrix(rows, cols, filename)
print(f"Random float matrix of size {rows}x{cols} has been written to {filename}")

sum_matrix = mat1 + mat2 + mat3

# Print the result
print("Summed Matrix:")
total_sum = np.sum(sum_matrix)
print(total_sum)   

