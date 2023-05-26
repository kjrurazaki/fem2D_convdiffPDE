import numpy as np


def imposeBC(model, method, penalty):
    """
    Impose Dirichlet BCs
    """
    stiffMat = model.stiffMat.copy()

    rhs = np.zeros((1, model.Nodes))
    u_dir = np.zeros((model.Nodes, 1))
    index_nodes = list(range(0, model.Nodes))
    boundary_nodes = []
    if method == "penalty":
        for idir in range(0, model.NDir):
            iglob = model.DirNod[idir]
            boundary_nodes.append(iglob - 1)
            stiffMat[iglob - 1, iglob - 1] = penalty
            rhs[0, iglob - 1] = penalty * model.DirVal[idir]
    elif method == "lifting":
        for idir in range(0, model.NDir):
            iglob = model.DirNod[idir]
            index_nodes.remove(iglob - 1)
            boundary_nodes.append(iglob - 1)
            u_dir[iglob - 1, 0] = model.DirVal[idir]
        for inod in index_nodes:
            rhs[0, inod] = -1 * np.matmul(stiffMat[inod, :], u_dir)
        for iglob in boundary_nodes:
            stiffMat[iglob, :] = 0
            stiffMat[:, iglob] = 0
            stiffMat[iglob, iglob] = 1
            rhs[0, iglob] = u_dir[iglob, 0]
    return stiffMat, rhs, u_dir, index_nodes, boundary_nodes
