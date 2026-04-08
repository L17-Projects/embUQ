***************************************
Tau Leaping Algorithm
***************************************

This models implements the TauLeaping algorithm, as published in `Cao2005 <https://aip.scitation.org/doi/pdf/10.1063/1.1992473>`_.

The tau-leaping algorithm speeds up the SSA algorithm by approxiamting the number of firings for each reaction channel during a chosen time increment (tau) as a Poisson variable. This algorithm avoids negative reactants using a simple trick and it is generally more accurate thant other variants using a Poisson procedure.
