from ed_lgt.workflows.su2 import compare_SU2_models
import logging

logger = logging.getLogger(__name__)


# Boundary conditions: OBC vs PBC per dimension
def obc_flags(dim, obc: bool):
    return [obc] * dim


# Optionally tune neigs by dimension to keep runtime low
def choose_neigs(dim):
    return 4 if dim <= 2 else 2  # tweak if needed


def main():
    atol = 1e-10
    g, m = 0.1, 0.1
    cases = []
    # 1D: only pure_theory=False
    cases += [dict(lvals=[4], dim=1, pure_theory=False)]
    # 2D
    cases += [
        dict(lvals=[2, 2], dim=2, pure_theory=False),
        dict(lvals=[2, 2], dim=2, pure_theory=True),
    ]
    # 3D
    cases += [
        dict(lvals=[2, 2, 2], dim=3, pure_theory=False),
        dict(lvals=[2, 2, 2], dim=3, pure_theory=True),
    ]
    n_passed = 0
    for case in cases:
        lvals = case["lvals"]
        dim = case["dim"]
        pure_theory = case["pure_theory"]
        neigs = choose_neigs(dim)
        bc_names = ["OBC", "PBC"] if dim < 3 else ["OBC"]
        bc_bools = [True, False] if dim < 3 else [True]
        for kk, bc_case in enumerate(bc_names):
            has_obc = obc_flags(dim, bc_bools[kk])
            logger.info("****************************************************")
            logger.info("")
            logger.info(f"Testing SU2 model: dim={dim}, lvals={lvals}")
            logger.info(f"bc={bc_case}, pure={pure_theory}, neigs={neigs}")
            logger.info("")
            logger.info("****************************************************")
            try:
                compare_SU2_models(
                    lvals=lvals,
                    pure_theory=pure_theory,
                    has_obc=has_obc,
                    g=g,
                    m=m,
                    atol=atol,
                    neigs=neigs,
                )
            except Exception as e:
                raise AssertionError(
                    f"FAIL std vs gen (spin=1/2): "
                    f"dim={dim}, lvals={lvals}, bc={bc_case}, pure={pure_theory}, "
                    f"neigs={neigs}, g={g}, m={m}. "
                    f"Error: {e}"
                ) from e
            logger.info("****************************************************")
            logger.info("")
            logger.info("TEST PASSED")
            logger.info("")
            logger.info("****************************************************")
            n_passed += 1
    logger.info("====================================================")
    logger.info(f"ALL PASSED ({n_passed} cases).")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
