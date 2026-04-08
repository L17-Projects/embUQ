***************
Nested Sampling
***************

This is an implementation of the *Nested Sampling* by John Skilling,
as published in `https://projecteuclid.org/euclid.ba/1340370944`.

The implementation of the *Multi Ellipse* proposal distribution is based on
the work of Feroz et. al. `https://academic.oup.com/mnras/article/398/4/1601/981502`.

Our version of the Multi Nest algorithm include a pior repartitioning strategy `https://link.springer.com/article/10.1007/s11222-018-9841-3`  to efficiently sample unrepresentative priors.
