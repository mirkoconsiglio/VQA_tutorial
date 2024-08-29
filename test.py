# General imports
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp, Statevector

# BFGS minimizer routine
from qiskit_algorithms.optimizers import L_BFGS_B

# Plotting functions
import matplotlib.pyplot as plt

def cost_function(theta, ansatz, hamiltonian):
	circuit = ansatz.assign_parameters(theta)  # bind parameters to circuit
	statevector = Statevector(circuit)  # get statevector representation
	exp = statevector.expectation_value(hamiltonian).real  # evaluation expectation value

	return exp


hamiltonian = SparsePauliOp.from_list(
    [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
)

ansatz = EfficientSU2(hamiltonian.num_qubits)
ansatz.decompose().draw("mpl", style="iqp")

optimizer = L_BFGS_B(args=(ansatz, hamiltonian))  # extra arguments in the cost function need to be included

x0 = np.random.uniform(-np.pi, np.pi, len(ansatz.parameters))  # We need an initial point (which can be random)

result = optimizer.minimize(fun=cost_function, x0=x0)

print(result)
