# Simulation of VGLUT1 channel function or active transport

We provide kinetic models, including ensembles of parameters optimised against electrophysiological recordings, to simulate various functions of WT and H120A VGLUT1 membrane proteins as described in our publication. Time courses of VGLUT1 Cl<sup>-</sup> channel open probabilities are simulated for the WT and H120A mutant protein, via a channel function. Secondary-active glutamate or aspartate transport current is simulated as the charge flux between states based on transport cycles.

The `scripts` folder contains the code used for our analysis for reproducibility.

## Inputs

Ligand concentrations, such as internal or external pH and [Cl<sup>-</sup>], should be provided together with V as the conditions, in addition to a list containing all rate constants and other variables needed (k1, k2, z, d) to calculate flux. Variable parameters describing state-state interactions are available in the pkl files provided for each kinetic model. For each of these two models, a function that contains and applies the appropriate model framework based on given conditions, is provided along with a pkl file containing ensembles of kinetic parameters that specify the interactions between model states.

* [Cl_WT_parms.pkl](./Cl_conduction_sim/Cl_WT_parms.pkl) contains >10000 parameter sets with >3000 variants for each of the 80 parameters.

* [Cl_H120A_parms.pkl](./Cl_conduction_sim/Cl_H120A_parms.pkl) contains >10000 parameter sets with >3000 variants for each of the 80 parameters.

* [Glut_Asp_parms.pkl](./GA_exchange_sim/Glut_Asp_parms.pkl) contains >20000 parameter sets with >980 variants for each of the 164 parameters.

## Code usage

[Cl_openprob.py](./Cl_conduction_sim/Cl_openprob.py) simulates the change in open probability upon perturbation at t=0s from an initial condition to a target condition.<br>
[GA_current.py](./GA_exchange_sim/GA_current.py) simulates the current time course upon perturbation at t=0s from an initial condition to a target condition.

```
Input:

usage: Cl_openprob.py [-h] -f F -pHext0 PHEXT0 -pHext1 PHEXT1 -pHint0 PHINT0 -pHint1 PHINT1 -Clext0 CLEXT0 -Clext1 CLEXT1 -Clint0 CLINT0 -Clint1
                      CLINT1 -V0 V0 -V1 V1 -nsteps NSTEPS -freq FREQ [-out OUT] [-n N]

options:
  -h, --help      show this help message and exit.
  -f              Path for the input parameters. (Default Cl_WT_parms.pkl)
  -pHext0         External pH of the initial condition. (Default 5.5)
  -pHext1         External pH of the target condition. (Default 5.5)
  -pHint0         Internal pH of the initial condition. (Default 7.4)
  -pHint1         Internal pH of the target condition. (Default 7.4)
  -Clext0         External [Cl-] in M of the initial condition. (Default 0.140)
  -Clext1         External [Cl-] in M of the target condition. (Default 0.140)
  -Clint0         Internal [Cl-] in M of the initial condition. (Default 0.140)
  -Clint1         Internal [Cl-] in M of the target condition. (Default 0.140)
  -V0             Membrane voltage in volts of the initial condition. (Default 0)
  -V1             Membrane voltage in volts of the target condition. (Default -0.160)
  -nsteps         Number of time steps. The total time, in s, is nsteps/freq. (Default 3000)
  -freq           Frequency of datapoints per second in Hz. (Default 100000)
  -out            Output file in txt format. The file contains two columns, the first representing the time in seconds and the second the open
                  probability. (Default sim.txt)
  -n              Specify which set of parameters to select in the input file. (Default 0)
```

[GA_current.py](./GA_exchange_sim/GA_current.py) calculates the integrated pathway between two transition matrices (TM1 and TM2, built from the model framework, kinetic parameters, and distinct given conditions) as charge flux, for a specified amount of time in seconds represented by and calculated as nsteps / freq.

```
Input:

usage: GA_current.py [-h] -f F -sub {G,A} -pHext0 PHEXT0 -pHext1 PHEXT1 -pHint0 PHINT0 -pHint1 PHINT1 -Clext0 CLEXT0 -Clext1 CLEXT1 -Sint0 SINT0
                     -Sint1 SINT1 -V0 V0 -V1 V1 -tsteps TSTEPS -freq FREQ [-out OUT] [-n N]

options:
  -h, --help      show this help message and exit.
  -f              Path for the input parameters. (Default Glut_Asp_parms.pkl)
  -sub {G,A}      Select the substrate to consider (Glutamate (G) or Aspartate (A), default A)
  -pHext0         External pH of the initial condition. (Default 5.5)
  -pHext1         External pH of the target condition. (Default 5.5)
  -pHint0         Internal pH of the initial condition. (Default 7.4)
  -pHint1         Internal pH of the target condition. (Default 7.4)
  -Clext0         External [Cl-] in M of the initial condition. (Default 0)
  -Clext1         External [Cl-] in M of the target condition. (Default 0.04)
  -Sint0          Internal substrate concentration (in M) of the initial condition. (Default 0.140)
  -Sint1          Internal substrate concentration (in M) of the target condition.(Default 0.140)
  -V0             Membrane voltage in volts of the initial condition. (Default -0.160)
  -V1             Membrane voltage in volts of the target condition. (Default -0.160)
  -nsteps         Number of time steps. The total time, in s, is nsteps/freq. (Default 2000)
  -freq           Frequency of datapoints per second in Hz. (Default 5000)
  -out            Output file in txt format. The file contains two columns, the first representing the time in seconds and the second the total
                  current in terms of unitary charge per second. (Default sim.txt)
  -n              Specify which set of parameters to select in the input file. (Default 0)
```

# Construction of the transition matrix

## Flux values

Transitionmatrix is a function that fills a 2D numpy matrix with rate flux values.
For each two connected states, the flux between them is based on 4 constants, for e.g. connection x:

     forward rate constant kx1 (from state A to state B)
     reverse rate constant kx2 (from state B to state A)
     charge movement zx (-1 to 1)
     symmetry factor dx (0 to 1)

From these constants, two net rates are calculated:

     k0x = kx1 * np.exp( zx *    dx  * F * V/(R * T))
     kx0 = kx2 * np.exp(-zx * (1-dx) * F * V/(R * T))

It makes sense to use a convention to distinguish <kx1 & k0x> from <kx2 & kx0> here, e.g., <kx1 & k0x> always being binding steps and conformation changes away from the conformation of the first state.

## A matrix of state connections

The number of states determines the size of the transition matrix, as len(states) by len(states).
Each row in the matrix represents a state, with each column representing the state from which flux enters it.
For state B, matrix index [B, A] contains the flux from state A as e.g. k0x ( * ligand concentration, if applicable).
The reverse process, kx0 from B to A, should be in index [A, B]. 
All diagonal indices, i.e. index [A, A], represent the sum of outward (negative) fluxes as e.g. -k0x -ky0.
