import os
import pickle
import numpy as np
from .checks import validate_parameters

__all__ = [
    "save_dictionary",
    "load_dictionary",
    "save_data_in_textfile",
    "load_data_from_textfile",
    "save_sparse_matrix_to_dat",
]


def save_dictionary(dictionary, filename):
    """
    This function save the information of a Python dictionary into a .pkl file

    Args:
        dictionary (dict): dictionary to be saved

        filename (str): name of the file where to save the dictionary
    """
    # Validate type of parameters
    validate_parameters(dictionary=dictionary, filename=filename)
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dictionary, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


def load_dictionary(filename):
    """
    This function loads the information of a Python dictionary from a .pkl file

    Args:
        filename (str): name of the file where the dictionary is saved
    """
    # Validate type of parameters
    validate_parameters(filename=filename)
    with open(filename, "rb") as outp:
        return pickle.load(outp)


def save_data_in_textfile(data_file, x_data, new_data):
    """
    This function stores a set of values as a new column of a text file with already existing columns of values.
    Each column will be then easily used for comparison in plots.

    Args:
        data_file (str): Name of the file where to save the set of values

        x_data (list): It contains the x values corresponding to the new set of values.
            The first entry is a string label

        new_data (list): It contains the new set of y values corresponding to the x_data.
            The first entry is a string label (typicalliy referred to the simulation)

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.
    """
    if not isinstance(data_file, str):
        raise TypeError(f"data_file should be a STRING, not a {type(data_file)}")
    if not isinstance(x_data, list):
        raise TypeError(f"x_data must be a LIST, not a {type(x_data)}")
    if not isinstance(new_data, list):
        raise TypeError(f"new_data must be a LIST, not a {type(new_data)}")
    # STORE X VALUES
    if not os.path.exists(data_file):
        g = open(data_file, "w+")
        for ii in range(len(x_data)):
            g.write(str(x_data[ii]) + "\n")
        g.close()
    # STORE NEW Y VALUES
    f = open(data_file, "r+")
    line = f.readlines()
    f.close()
    h = open(data_file, "w+")
    for ii in range(len(line)):
        line[ii] = line[ii].rstrip()
        h.write(line[ii] + "," + str(new_data[ii]) + "\n")
    h.close()


def load_data_from_textfile(data_file_name, row_for_labels=False):
    """
    This function acquires data from a text file made out of different columns and yields it as a dictonary

    Args:
        data_file_name (str): name of the file

        row_for_labels (bool, optional): If True, the firs line contains the labels. Default to False.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        dict: Dictonary where all the informations are stored
    """
    if not isinstance(data_file_name, str):
        raise TypeError(
            f"data_file_name should be a STRING, not a {type(data_file_name)}"
        )
    if not isinstance(row_for_labels, bool):
        raise TypeError(f"row_for_labels must be a BOOL, not a {type(row_for_labels)}")
    # Open the file and acquire all the lines
    f = open(data_file_name, "r+")
    line = f.readlines()
    f.close()
    # CREATE A DICTIONARY TO HOST THE LISTS OBTAINED FROM EACH COLUMN OF data_file
    data = {}
    # Get the first line of the File as a list of entries.
    n = line[0].strip().split(",")
    for ii in range(0, len(n)):
        # Generate a list for each column of data_file
        data[str(ii)] = list()
        if row_for_labels:
            # Generate a label for each list acquiring the ii+1 entry of the first line n
            data["label_%s" % str(ii)] = str(n[ii])

    if row_for_labels:
        # IGNORE THE FIRST LINE OF line (ALREAY USED FOR THE LABELS)
        del line[0]

    # Fill the lists with the entries of Columns
    for ii in range(len(line)):
        a = line[ii].strip().split(",")

        for jj in range(len(n)):
            data[str(jj)].append(float(a[jj]))
    for ii in range(len(n)):
        data[str(ii)] = np.asarray(data[str(ii)])
    return data


def save_sparse_matrix_to_dat(sparse_matrix, filename):
    """
    Save the non-zero entries of a scipy.sparse.csr_matrix to a .dat file.

    Parameters:
    A (scipy.sparse.csr_matrix): The sparse matrix to save.

    filename (str): The name of the file where the matrix will be saved.
    """
    validate_parameters(op_list=[sparse_matrix], filename=filename)

    with open(filename, "w") as f:
        # Write the dimension of the matrix
        f.write("# dimension\n")
        f.write(f"{sparse_matrix.shape[0]}\n")
        # Write the non-zero elements
        f.write("# Non-zero elements: coordinates and coefficients\n")
        coo = sparse_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            f.write(f"{i}, {j}; ({v.real}, {v.imag})\n")
