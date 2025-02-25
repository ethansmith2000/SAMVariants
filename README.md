# Sharpness Aware Minimzation Variants

The original sharpness aware minimization involves producing a gradient as normal, as well as from a perturbed position that is the steepest loss increase, such that we can incentivize converging to points where the lanscape is not overly sharp. The cost unfortunately makes its usage hard to justify. MomentumSAM proposes avoiding needing to compute gradient more than once by using a perturbation based on the momentum term. Its results are okay but leave more to be desired. Gallabytes suggested performing the perturbation based on another optimizers update direction. This repo implements that, namely thus far using Adam/Muon combinations. There are also other experimental versions using weight decay or dual momentum terms for the perturbation.


reference papers:
- https://arxiv.org/abs/2010.01412
- https://arxiv.org/abs/2401.12033

