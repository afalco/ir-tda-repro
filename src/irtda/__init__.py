"""
irtda -- Topological characterisation of composite materials from IR/ATR spectra.

The package implements the persistent-homology workflow described in

    T. Frahi, C. Argerich, M. Yun, A. Falco, A. Barasinski, F. Chinesta,
    "Tape surfaces characterization with persistence images",
    AIMS Materials Science 7(4):364-380, 2020.

    T. Frahi, A. Falco, B. Vinh Mau, J. L. Duval, F. Chinesta,
    "Empowering Advanced Parametric Modes Clustering from Topological Data
    Analysis", Applied Sciences 11:6554, 2021.

    T. Frahi, A. Sancarlos, M. Galle, X. Beaulieu, A. Chambard, A. Falco,
    E. Cueto, F. Chinesta, "Monitoring Weeder Robots and Anticipating Their
    Functioning by Using Advanced Topological Data Analysis",
    Frontiers in Artificial Intelligence 4:761123, 2021.

adapted to one-dimensional infrared absorbance curves.
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    clustering,
    dataset,
    features,
    images,
    persistence,
    plotting,
)

__all__ = [
    "clustering",
    "dataset",
    "features",
    "images",
    "persistence",
    "plotting",
]
