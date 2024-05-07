# %%
from simsio import *
from matplotlib import pyplot as plt

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


# %%
config_filename = f"QED/scaling_conv"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "spin"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["spin"]))),
}

for ii, g in enumerate(vals["g"]):
    for kk, spin in enumerate(vals["spin"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["E_square"][ii, kk] = np.mean(get_sim(ugrid[ii][kk]).res["E_square"])
        res["plaq"][ii, kk] = get_sim(ugrid[ii][kk]).res[
            "C_px,py_C_py,mx_C_my,px_C_mx,my"
        ]
abs_convergence = 1e-6
rel_convergence = 1e-7
res["convergence"] = np.zeros((len(vals["g"])))
for ii, g in enumerate(vals["g"]):
    for kk, spin in enumerate(vals["spin"]):
        if kk > 0:
            abs_delta = np.abs(res["plaq"][ii, kk] - res["plaq"][ii, kk - 1])
            rel_delta = abs_delta / np.abs(res["plaq"][ii, kk])
            if abs_delta < abs_convergence and rel_delta < rel_convergence:
                print(g, spin)
                res["convergence"][ii] = spin
                break
print("===========================")
for ii, g in enumerate(vals["g"]):
    if res["convergence"][ii] == 0:
        res["convergence"][ii] = 30

prefactor = 2.9
fig, ax = plt.subplots()
ax.plot(
    1 / vals["g"],
    res["convergence"],
    "-o",
    label=r"$j_{\min} (\Delta_{re\ell} =10^{-7}, \Delta_{abs} =10^{-6})$",
)
ax.set(
    xscale="log",
    yscale="log",
    ylim=[3, 50],
    xlim=[3e-1, 10],
    xlabel="1/g",
    ylabel=r"$j_{\min}$",
)
ax.plot(1 / vals["g"], prefactor / vals["g"], label=r"$3*g^{-1}$")
ax.legend()
plt.savefig("QED_conv.pdf")
# %%
