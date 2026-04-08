*******************************
Distributed Conduit
*******************************

This distributed conduit uses MPI to distribute sample evaluation among *n* workers. Each worker consists of *k* MPI ranks, where *k* is a configurable parameter. Communication among workers is realized via MPI messages.

This model is ideal for when your computational model can be directly linked with Korali and/or expects an MPI communicator itself. 

For an example on how to create a MPI/Python Korali application, see: :ref:`MPI/Python Example <feature_running.mpi.python>`).
For an example on how to create a MPI/C++ Korali application, see: :ref:`MPI/C++ Example <feature_running.mpi.cxx>`). 
For more information, see :ref:`Parallel Execution <parallel-execution>`. 

