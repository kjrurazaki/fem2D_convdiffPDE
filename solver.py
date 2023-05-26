from scipy.sparse.linalg import gmres, spilu, LinearOperator, splu


def gmres_solver(SYSMAT, RHS):
    # Set parameters
    restart = 10
    tol = 1e-9
    maxit = 20

    try:
        # Try using SPILU preconditioner
        ilu = spilu(SYSMAT, fill_factor=1)
        M = LinearOperator((SYSMAT.shape[0], SYSMAT.shape[1]), ilu.solve)
        error = 0
    except RuntimeError as e:
        # If singularity error occurs, use alternative preconditioner
        print("Error:", e)
        print("Using alternative preconditioner...")
        lu = splu(SYSMAT)
        M = LinearOperator((SYSMAT.shape[0], SYSMAT.shape[1]), lu.solve)
        error = "singular"

    # Solve the system using GMRES with the preconditioner
    x, info = gmres(SYSMAT, RHS, restart=restart, tol=tol, maxiter=maxit, M=M)
    return x, info, error
