***************************************
SSA (Stochastic Simulation Algorithm)
***************************************

`The Gillespie algorithm (1977) <https://pubs.acs.org/doi/10.1021/j100540a008>`_ simulates trajectories of a stochastic equation system (reactions) for which the rates are known. If the number of reactants is high, the time steps become small and the algorithm stalls. This behaviour is alleviated in the tau-leaping algorithm.
