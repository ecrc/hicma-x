import numpy as np
import os

def read_and_concatenate_binary_files(directory_path, output_file_path, dtype=np.float32, num_columns=10):
    """
    Reads binary files from the specified directory, concatenates them along columns, and writes the result to a binary file.

    :param directory_path: Path to the directory containing the binary files.
    :param output_file_path: Path to save the concatenated binary file.
    :param dtype: Data type of the binary data (default is np.float32).
    :param num_columns: Number of columns to reshape each binary file into (must match the structure of your data).
    """
    # List all binary files in the directory
    files = [file for file in os.listdir(directory_path) if file.endswith('.bin')]

    # Initialize a list to store arrays
    arrays = []

    # Loop over each file, read it, and append the array to the list
    for file in files:
        file_path = os.path.join(directory_path, file)

        # Read the binary file into a 1D array
        data = np.fromfile(file_path, dtype=dtype)

        # Reshape the data to a 2D array with the specified number of columns
        try:
            data_reshaped = data.reshape((-1, num_columns))
            print(data_reshaped.shape)
            arrays.append(data_reshaped)
        except ValueError:
            print(f"Skipping file {file} due to incompatible shape")

    # Concatenate arrays along the columns (axis 1)
    concatenated_array = np.concatenate(arrays, axis=1)
    print(concatenated_array)
    print(concatenated_array.shape)
    # Write the concatenated array to a binary file
    concatenated_array.tofile(output_file_path)

    print(f"Concatenated data written to {output_file_path}")

# Example usage
directory_path = '.'
output_file_path = 'concatenated_data80.bin'
read_and_concatenate_binary_files(directory_path, output_file_path, dtype=np.float32, num_columns=10000)
