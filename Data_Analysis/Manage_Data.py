import numpy as np
import json
import os


def store_results(data_file, x_data, new_data):
    if not isinstance(data_file, str):
        raise TypeError(f"data_file should be a STRING, not a {type(data_file)}")
    if not isinstance(x_data, list):
        raise TypeError(f"x_data must be a LIST, not a {type(x_data)}")
    if not isinstance(new_data, list):
        raise TypeError(f"new_data must be a LIST, not a {type(new_data)}")
    """
    THIS FUNCTION STORES SOME VALUES IN NEW COLUMN OF A FILE
    WITH ALREADY EXISTING COLUMNS OF vocabulary. THESE WOULD
    BE THEN EASILY COMPARED IN A PLOT, ALL TOGETHER.
    *    data_file   IS THE NAME OF THE FILE WHERE ALL THE
    *                ALREADY EXISTING SIMULATIONS LIE
    *    x_data      LIST CONTAINING X VALUES (FIRST ENTRY = LABEL OF x AXIS)
    *    new data    IS A LIST CONTAINING ALL THE NEW DATA (y VALUES).
    *                THE FIRST ELEMENT OF THE LIST IS A STRING
    *                LABELING THE NAME OF THE SIMULATION
    """
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


def save_dictionary(dict, filename):
    with open(f"{filename}.json", "w") as outp:
        json.dump(dict, outp, indent=4)
    outp.close


def acquire_data(data_file_name, row_for_labels=False):
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
