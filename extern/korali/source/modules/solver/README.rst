**************************
Solver
**************************

Solver modules are algorithms/method to obtain the solution to a particular :ref:`problem <module-problem>`. Solvers are selected in each experiment $e$ with the following syntax:
 
.. code-block:: python

   e = korali.Experiment()
   e["Solver"]["Type"] = "Optimizer/CMAES"  
   
For more information, see :ref:`Korali Usage Basics <basics>`. 