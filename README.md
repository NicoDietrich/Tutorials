# Optimization and Gradient Verification for multiphysics/turb_rht_cht/

The added scripts in scripts.py allow for comparison of the gradients provided by SU2 with a central difference approximation.
I tried to evaluate the perturped functional in parallel by using multiple
config files and starting multiple mpi procesesses but ran into problems
(shared binaries?) I haven't tried to solve yet.
Additionally an adapted armijo rule is implemented to test possible applications of the gradients in a more complex setting.

## Dependencies
The scripts are tested with **SU2 7.0.6.** but should work with newer versions as
well. To run SU2 **openmpi** is used. To quickly parse output files standard
linux tools (**tail**, **awk**) are used.

## Differences

Note that the definition of the ffd box differs is comparison to the upstream repository, as the inclusion of the upper left meshpoint resulted in a broken mesh.

Checkout https://su2code.github.io/tutorials/Turbulent_RHT_CHT/ for a detailed explanation of the SU2 part of the program.
