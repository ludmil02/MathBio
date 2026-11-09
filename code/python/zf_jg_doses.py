"""Vaccine dose budgets b1, b2, b3, the right hand sides of the chi constraints.

The constraints are, per location i,

    sum_j chi_k[i, j] X_k[i, j] = b_k[i],   k = 1, 2, 3,

with X_1 = S, X_2 = V, X_3 = P at the end of the previous week.

b1 comes from C2 of 16_Group_Model_Results_t=347_ZF.nb, the doses of the
primary series by location and age group; b1[i] is the row sum, which is what
that notebook used to export as b1Vector.csv.  No booster budget has ever been
given, so b2 and b3 are zero.

Nothing here is read from or written to disk.
"""

import numpy as np

# doses by location (Northeast, Midwest, South, West) and age group
# (0-19, 20-39, 40-59, 60+); C2 of 16_Group_Model_Results_t=347_ZF.nb
C2 = np.array([[279, 62294, 60028, 24269],
               [898, 92762, 96407, 37373],
               [1180, 115330, 133603, 56325],
               [749, 107797, 119699, 39046]], dtype=float)


def dose_budgets(c2=None):
    """Return (b1, b2, b3), each of length m.

    b1 = row sums of c2, i.e. Total[Transpose[C2]] of the notebook.
    b2 = b3 = 0, no booster doses being specified.
    """
    c2 = C2 if c2 is None else np.asarray(c2, dtype=float)
    b1 = c2.sum(axis=1)
    return b1, np.zeros_like(b1), np.zeros_like(b1)


if __name__ == "__main__":
    b1, b2, b3 = dose_budgets()
    print("b1 =", b1)
    print("b2 =", b2)
    print("b3 =", b3)
