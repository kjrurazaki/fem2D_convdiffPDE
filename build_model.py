from localBasis import localBasis
from stiffBuild import stiffBuild

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt


class Model:
    def __init__(self, meshdir, b):
        self.meshdir = meshdir
        self.load_data()
        self.build_basis()
        self.b = np.array(self.Nelem * b).reshape(-1, 2)  # Velocity
        self.delta = None  # Numerical diffusion
        self.build_model()
        self.uh = None
        self.peclet = None

    def load_data(self):
        """
        Define the 2D mesh and equation BC's and coefficients
        """
        _mesh = np.genfromtxt(self.meshdir + "/mesh.dat", dtype=int)
        self.coord = np.genfromtxt(self.meshdir + "/xy.dat")
        self.DirNod = np.genfromtxt(self.meshdir + "/DirNod.dat", dtype=int)
        self.DirVal = np.genfromtxt(self.meshdir + "/DirVal.dat")

        # Remove indices of elements
        self.triang = _mesh[:, 0:3]
        self.coord = self.coord[:, :]
        self.DirNod = self.DirNod.reshape(-1, 1)
        self.DirVal = self.DirVal.reshape(-1, 1).astype(float)
        self.Diff = _mesh[:, 3].reshape(-1, 1).astype(float)
        self.Nelem = len(self.triang)
        self.Nodes = len(self.coord)
        self.NDir = len(self.DirNod)

    def build_model(self):
        """
        Build basis functions, stiffness matrix and rhs
        """
        # build stiffness matrix (without BCs)
        self.diffMat, self.transportMat, self.numdiffMat, self.stiffMat = stiffBuild(
            self
        )

    def build_basis(self):
        # Calculate element area and elemental coeffients of basis functions
        # (b,c)
        self.Aloc, self.Bloc, self.Cloc, self.Area, self.h = localBasis(self)

    def update_diff(self, value_diff):
        self.Diff[:] = value_diff
        self.build_model()

    def update_delta(self, value_delta):
        self.delta = value_delta
        self.build_model()

    def update_concentration(self, value_conc):
        self.uh = value_conc

    def print_parameters(self):
        """
        Print parameters
        """
        print(
            self.Nelem,
            "\n",
            self.Nodes,
            "\n",
            self.NDir,
            "\n",
            self.triang[0:5, :],
            "\n",
            self.coord[0:5, :],
            "\n",
            self.DirNod[0:5, :],
            "\n",
            self.DirVal[0:5, :],
            "\n",
            self.Diff[0:5, :],
            "\n",
        )
        print(self.b[0:5, :], "\n")

        print(
            self.Aloc[0:5, :],
            "\n",
            self.Bloc[0:5, :],
            "\n",
            self.Cloc[0:5, :],
            "\n",
            self.Area[0:5, :],
            "\n",
            self.h[0:5, :],
        )

        print(
            self.stiffMat[0:5, :5],
            "\n",
            self.diffMat[0:5, :5],
            "\n",
            self.numdiffMat[0:5, :5],
            "\n",
            self.transportMat[0:5, :5],
            "\n",
        )

    def convective_flow(self):
        df = self.sides_dataframe()
        df["mass_flow"] = df.apply(
            lambda a: np.matmul(self.b[0, :], a["Outer Normal"])
            * a["Concentration"]
            * a["Side Length"] ** 2,
            axis=1,
        )
        return df, -1 * sum(df["mass_flow"])

    def diffusion_flow(self):
        boundary_nodes = self.DirNod - 1
        return sum(np.matmul(self.diffMat, self.uh)[boundary_nodes])[0][0]

    def numerical_diffusion_flow(self):
        boundary_nodes = self.DirNod - 1
        if self.numdiffMat.shape[1] == 0:
            return None
        else:
            return sum(np.matmul(self.numdiffMat, self.uh)[boundary_nodes])[0][0]

    def sides_dataframe(self):
        # Define the sides of the square and their corresponding regions
        sides = ["1", "2", "3", "4", "5"]
        regions = ["Γ1", "Γ1", "Γ2", "Γ2", "Γ2"]

        # Define the side lengths and outer normals for each side
        side_lengths = [1, 0.3, 1, 1, 0.7]
        outer_normals = [[-1, 0], [0, -1], [0, 1], [1, 0], [0, -1]]
        concentration_value = [1, 1, 0, 0, 0]

        # Create the dataframe
        df = pd.DataFrame(
            {
                "Region": regions,
                "Side Length": side_lengths,
                "Outer Normal": outer_normals,
                "Concentration": concentration_value,
            },
            index=sides,
        )
        return df

    def plot_boundary(self):
        # Define the vertices of the square
        vertices = np.array([(0, 0), (0.3, 0), (1, 0), (1, 1), (0, 1)])

        # Define the sides of the square and their corresponding regions
        sides = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

        regions = ["Γ1", "Γ2", "Γ2", "Γ2", "Γ1"]

        # Define the colors for each region
        colors = {"Γ1": "blue", "Γ2": "red"}

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the sides of the square and color each region
        for i, side in enumerate(sides):
            print(side)
            print(vertices[side, :])
            x, y = zip(*vertices[side, :])
            region = regions[i]
            color = colors[region]
            ax.plot(x, y, color=color, linewidth=2)

        # Set the axis limits and labels
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Add a legend for the regions
        handles = [
            plt.plot([], [], color=color, linewidth=2)[0] for color in colors.values()
        ]
        labels = list(colors.keys())
        ax.legend(handles, labels)
