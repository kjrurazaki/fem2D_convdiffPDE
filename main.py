# Finite element code for 2D piecewise Linear Galerkin
# Convection - difussion PDE
# -div(Diff grad u)(x) + div(b u) = f(x) with Dirichlet BCs
# Velocity constant div(b) = 0
# Forcing f(x) = 0
# Only Dirichlet boundary conditions

from solver import gmres_solver
from imposeBC import imposeBC

from build_model import Model

import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
import time

from matplotlib import pyplot as plt

import display_results


def run_2D(model, method, penalty):
    # Impose BCs
    stiffMat, rhs, u_dir, index_nodes, boundary_nodes = imposeBC(
        model, method=method, penalty=penalty
    )

    # Concentration vector
    uh = np.zeros((model.Nodes, 1))

    # Convert the numpy array to a sparse matrix
    sparse_stiffMat = csc_matrix(stiffMat)

    x, info, e = gmres_solver(sparse_stiffMat, rhs.reshape(-1, 1))
    # print(info)
    uh = x.reshape(-1, 1)

    # uh = uh + u_dir
    # Residual
    residual = np.linalg.norm(
        np.zeros((1, model.Nodes)) - np.matmul(model.stiffMat, uh)
    )
    residual_bc = np.linalg.norm(u_dir[boundary_nodes, :] - uh[boundary_nodes, :])
    return uh, residual, residual_bc, info, e


def grid_run(meshdir, b):
    """
    Generates results for many combination of parameters
    """
    global df_concentration
    global df_scenarios
    id = 0
    df_concentration = pd.DataFrame()
    df_scenarios = pd.DataFrame()

    list_delta = (
        [None] + list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1.01, 0.1))
    )
    list_diff = list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1.01, 0.1))
    list_penalty = (
        [10**i for i in range(0, 10, 1)]
        + [10**i for i in range(10, 16, 2)]
        + [10**i for i in range(30, 40)]
    )
    list_method = ["penalty", "lifting"]
    print(len(list_delta) * len(list_penalty) * len(list_method) * len(list_diff))

    def solve_append(id, model_convection_diff, method, penalty):
        global df_scenarios
        global df_concentration
        uh, residual, residual_bc, info, e = run_2D(
            model_convection_diff, method=method, penalty=penalty
        )

        model_convection_diff.update_concentration(uh)

        _df_conc = pd.DataFrame({"concentration": uh.flatten()})
        _df_conc["id"] = id
        df_concentration = pd.concat([df_concentration, _df_conc])

        _df_scena = pd.DataFrame(
            [
                {
                    "id": id,
                    "method": method,
                    "penalty": penalty,
                    "diff": model_convection_diff.Diff[0][0],
                    "delta": model_convection_diff.delta,
                    "velocity_x": model_convection_diff.b[0, 0],
                    "velocity_y": model_convection_diff.b[0, 1],
                    "h_max": max(model_convection_diff.h)[0],
                    "h_min": min(model_convection_diff.h)[0],
                    "diff_flow": model_convection_diff.diffusion_flow(),
                    "num_diff_flow": model_convection_diff.numerical_diffusion_flow(),
                    "convec_flow": model_convection_diff.convective_flow()[1],
                    "Peclet_triangle": model_convection_diff.peclet,
                    "residual_system": residual,  # residual of the system ||b - Ax||
                    "residual_bc": residual_bc,  # residual of Dirichlet BCs ||x_bc - x||
                    "convergence": info,
                    "error": e,
                }
            ]
        )
        df_scenarios = pd.concat([df_scenarios, _df_scena])

    model_convection_diff = Model(meshdir, b)
    for delta in list_delta:
        model_convection_diff.update_delta(delta)
        for diff in list_diff:
            model_convection_diff.update_diff(diff)
            for method in list_method:
                print(f"{diff},{method}, {delta}")
                if method == "penalty":
                    for penalty in list_penalty:
                        solve_append(id, model_convection_diff, method, penalty)
                        model_convection_diff.update_concentration(None)
                        id += 1
                        print(id)
                else:
                    solve_append(id, model_convection_diff, method, None)
                    id += 1
                print(id)

    df_concentration.to_csv("concentrations_id_2.csv")
    df_scenarios.to_csv("dim_id_residuals_2.csv")


if __name__ == "__main__":
    meshdir = "mesh"

    # penalty = 1E10
    # method = ['penalty', 'lifting'][0]
    # delta = None # Numerical diffusion, None == No numerical diffusion
    b = [1, 3]  # velocitys

    # model_convection_diff = Model(meshdir, b)
    # model_convection_diff.update_diff(0.01)
    # model_convection_diff.update_delta(delta)
    # # model_convection_diff.print_parameters()
    # uh, residual, residual_boundary, info, e = run_2D(model_convection_diff,
    #              method = method,
    #              penalty = penalty)
    # print(f'Residual:{residual}')
    # print(f'Residual BC: {residual_boundary}')
    # model_convection_diff.update_concentration(uh)
    # print(model_convection_diff.convective_flow()[0])
    # print(model_convection_diff.convective_flow()[1])
    # print(model_convection_diff.diffusion_flow())
    # print(model_convection_diff.numerical_diffusion_flow())
    # display_results.plot_field_3D(model_convection_diff.coord,
    #                               model_convection_diff.triang,
    #                               uh.flatten(),nodal=True,limit_z = False)
    # plt.show()

    grid_run(meshdir, b)
