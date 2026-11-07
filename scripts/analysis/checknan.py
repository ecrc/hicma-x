import numpy as np

def check_for_nan_in_binary_matrix(file_path):
    try:
        # Read the binary file
        data = np.fromfile(file_path, dtype=np.float32)
        
        # Check if there are any NaN values in the matrix
        has_nan = np.isnan(data).any()
        
        if has_nan:
            print("The matrix contains NaN values.")
        else:
            print("The matrix does not contain any NaN values.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = 'genotype5.bin'
check_for_nan_in_binary_matrix(file_path)

