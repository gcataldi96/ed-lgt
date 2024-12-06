# %%
import numpy as np
import pickle


def load_dictionary(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)


res = load_dictionary("SU3_2flavor_site.pkl")

# %%
print(res["operators"]["lnk_R"] - res["operators"]["lnk_L"])
# %%
