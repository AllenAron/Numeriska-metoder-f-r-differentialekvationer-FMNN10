import numpy as np
import matplotlib.pyplot as plt

def eulerstep(Tdx: np.ndarray, uold: np.ndarray, dt: float) -> np.ndarray:
    return np.add(uold,np.multiply(np.matmul(Tdx, uold), dt ))
