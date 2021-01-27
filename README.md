# Optimization and Gradient Verification for multiphysics/turb_rht_cht/

The added script finite_difference.py allows for comparison of the gradients provided by SU2 with a central difference approximation.
I tried to evaluate the perturped functional in parallel but ran into problems I haven't tried to solve yet.
Additionally an adapted armijo rule is implemented to test possible applications of the gradients in a more complex setting.

Note that the definition of the ffd box differs is comparison to the upstream repository, as the inclusion of the upper left meshpoint resulted in a broken mesh.
