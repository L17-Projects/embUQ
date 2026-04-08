**************************
Problems
**************************

Problem modules describe a statistical/learning/other problem to be solved by Korali. One or more :ref:`solvers <module-solver>` may exist (or be added to Korali) that can be used to solve a particular problem. To specify a problem type, use the following syntax:
 
.. code-block:: python

   e = korali.Experiment()
   e["Problem"]["Type"] = "Optimization"  
   
For more information, see :ref:`Korali Usage Basics <basics>`. 
