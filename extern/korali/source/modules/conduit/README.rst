*******************************
Conduits
*******************************

Conduit modules specify how the computational models used to evaluate samples are to be executed, in particular regarding parallelism/concurrency. The conduit to use is specified at the Korali Engine-level, and is shared among all the experiments in the run.

To select a specific conduit, use the following syntax:
 
.. code-block:: python

   k = korali.Engine()
   k["Conduit"]["Type"] = "Distributed"  
   
For more information, see :ref:`Parallel Execution <parallel-execution>`. 
