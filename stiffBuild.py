import numpy as np


def diffBuild(model):
    """
    Buid stiff matrix
    D_T - Diffusion coefficient of triangle
    a^T_{ij} = A_T * D_T * (b_i c_i) * (b_j c_j)
            = --- (b_i*b_j + c_i*c_j)
    """
    diffMat = np.zeros((model.Nodes, model.Nodes))
    for iel in range(0, model.Nelem):
        for iloc in range(0, 3):
            iglob = model.triang[iel, iloc]
            for jloc in range(0, 3):
                jglob = model.triang[iel, jloc]
                diffMat[iglob - 1, jglob - 1] = (
                    diffMat[iglob - 1, jglob - 1]
                    + (
                        model.Bloc[iel, iloc] * model.Bloc[iel, jloc]
                        + model.Cloc[iel, iloc] * model.Cloc[iel, jloc]
                    )
                    * model.Area[iel]
                    * model.Diff[iel]
                )
    return diffMat


def transportBuild(model):
    """
    Build transport matrix
    a^T_{i, j} = A_T * (B_x * b_j + B_y * c_j) * (a_i + 1/3 * sum_k_1^3[b_i * x_k + c_i * y_k])
    """
    transportMat = np.zeros((model.Nodes, model.Nodes))
    for iel in range(0, model.Nelem):
        p1 = model.coord[model.triang[iel, 0] - 1, :].reshape(
            1, -1
        )  # coordinate of first node
        p2 = model.coord[model.triang[iel, 1] - 1, :].reshape(
            1, -1
        )  # coordinate of second node
        p3 = model.coord[model.triang[iel, 2] - 1, :].reshape(
            1, -1
        )  # coordinate of third node
        for iloc in range(0, 3):
            Num_int = (
                model.Bloc[iel, iloc] * (p1[:, 0] + p2[:, 0] + p3[:, 0])
                + model.Cloc[iel, iloc] * (p1[:, 1] + p2[:, 1] + p3[:, 1])
            )[0] / 3
            iglob = model.triang[iel, iloc]
            for jloc in range(0, 3):
                jglob = model.triang[iel, jloc]
                transportMat[iglob - 1, jglob - 1] = (
                    transportMat[iglob - 1, jglob - 1]
                    + (
                        model.b[iel, 0] * model.Bloc[iel, jloc]
                        + model.b[iel, 1] * model.Cloc[iel, jloc]
                    )
                    * (model.Aloc[iel, iloc] + Num_int)
                    * model.Area[iel]
                )
    return transportMat


def sup_stabilization(model):
    """
    Streamline upwind stabilization
    a^T_{ij} = delta * A_T * Pe^T_h / |B|^2 *
                (b_i * b_j* B_x^2
               + c_i * c_j * B_y^2
               + (b_i * c_j + c_i * b_j) * B_x * B_y)
    """
    numdiffMat = np.zeros((model.Nodes, model.Nodes))
    for iel in range(0, model.Nelem):
        modB = np.sqrt(model.b[iel, 1] ** 2 + model.b[iel, 0] ** 2)
        Pe_t = modB * model.h[iel] / model.Diff[iel] / 2
        model.peclet = Pe_t
        for iloc in range(0, 3):
            iglob = model.triang[iel, iloc]
            for jloc in range(0, 3):
                jglob = model.triang[iel, jloc]
                numdiffMat[iglob - 1, jglob - 1] = numdiffMat[iglob - 1, jglob - 1] + (
                    model.Bloc[iel, iloc]
                    * model.Bloc[iel, jloc]
                    * (model.b[iel, 0] ** 2)
                    + model.Cloc[iel, iloc]
                    * model.Cloc[iel, jloc]
                    * (model.b[iel, 1] ** 2)
                    + (
                        model.Bloc[iel, iloc] * model.Cloc[iel, jloc]
                        + model.Bloc[iel, jloc] * model.Cloc[iel, iloc]
                    )
                    * model.b[iel, 0]
                    * model.b[iel, 1]
                ) * model.Area[iel] * model.delta * Pe_t / (modB**2)
    return numdiffMat


def stiffBuild(model):
    """
    Build stifness matrix
    """
    diffMat = diffBuild(model)
    transportMat = transportBuild(model)

    if model.delta is not None:
        numdiffMat = sup_stabilization(model)
        stiffMat = diffMat + transportMat + numdiffMat
    else:
        stiffMat = diffMat + transportMat
        numdiffMat = np.array([]).reshape(1, -1)
    return diffMat, transportMat, numdiffMat, stiffMat
