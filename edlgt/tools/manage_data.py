"""Utilities for saving and loading dictionaries, tabular text data, and sparse matrices.

This module provides lightweight I/O helpers used by examples and scripts:

- store Python dictionaries with ``pickle``,
- append data series to a simple comma-separated text file format,
- read tabular text files back into NumPy arrays,
- export sparse matrices to a human-readable ``.dat`` file.
"""

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
    """Serialize a Python dictionary to a pickle file.

    Parameters
    ----------
    dictionary : dict
        Dictionary to save.
    filename : str
        Output file path (typically with ``.pkl`` extension).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``dictionary`` or ``filename`` has an invalid type.
    """
    # Validate type of parameters
    validate_parameters(dictionary=dictionary, filename=filename)
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dictionary, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


def load_dictionary(filename):
    """Load a dictionary from a pickle file.

    Parameters
    ----------
    filename : str
        Path to a pickle file created by :func:`save_dictionary` (or compatible).

    Returns
    -------
    dict
        Deserialized dictionary.

    Raises
    ------
    TypeError
        If ``filename`` has an invalid type.
    """
    # Validate type of parameters
    validate_parameters(filename=filename)
    with open(filename, "rb") as outp:
        return pickle.load(outp)


def save_data_in_textfile(data_file, x_data, new_data):
    """Append a data series as a new column in a simple text table.

    The file format is line-based and comma-separated. On first write, the file
    is created and initialized with the values from ``x_data`` (usually an
    x-axis label followed by x values). On subsequent writes, each line gets one
    extra comma-separated entry from ``new_data``.

    Parameters
    ----------
    data_file : str
        Path to the text file to create/update.
    x_data : list
        Reference column written when the file does not yet exist.
        Conventionally the first item is a label and the remaining items are x
        values.
    new_data : list
        Column to append to the file. It must contain the same number of entries
        as the current file line count. Conventionally the first item is a label.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the input arguments have invalid types.

    Notes
    -----
    This helper assumes ``x_data`` and ``new_data`` are already aligned line by
    line. It does not validate lengths or parse values.
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
    """Load columnar numeric data from a comma-separated text file.

    Parameters
    ----------
    data_file_name : str
        Path to the input file.
    row_for_labels : bool, optional
        If ``True``, interpret the first line as column labels and store them as
        ``"label_0"``, ``"label_1"``, ... entries in the returned dictionary.
        Default is ``False``.

    Returns
    -------
    dict
        Dictionary containing one NumPy array per column under string keys
        ``"0"``, ``"1"``, ... . If ``row_for_labels`` is ``True``, label entries
        are also included.

    Raises
    ------
    TypeError
        If ``data_file_name`` or ``row_for_labels`` has an invalid type.

    Notes
    -----
    Data values are converted to ``float``. This helper expects a regular
    comma-separated table with the same number of columns on each line.
    """
    if not isinstance(data_file_name, str):
        raise TypeError(
            f"data_file_name should be a STRING, not a {type(data_file_name)}"
        )
    if not isinstance(row_for_labels, bool):
        raise TypeError(f"row_for_labels must be a BOOL, not a {type(row_for_labels)}")
    # Open the file and acquire all the lines
    f = open(data_file_name, "r")
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
    """Export a sparse matrix to a human-readable ``.dat`` text file.

    The output contains:

    - a header line with the matrix dimension,
    - one line per non-zero entry with row/column indices and complex value.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.spmatrix
        Sparse matrix to export.
    filename : str
        Output file path.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``sparse_matrix`` or ``filename`` has an invalid type.
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
