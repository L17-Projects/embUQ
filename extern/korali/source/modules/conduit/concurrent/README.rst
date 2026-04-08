*******************************
Concurrent Conduit
*******************************

This concurrent conduit uses fork/join mechanisms to distribute sample evaluation among *n* concurrent workers, each running as separate process from the main application process. Communication among workers is realized via OS pipes.

Use this model if your application cannot be parallelized with MPI or linked to Korali in any way.

For example, pre-packaged (black-box) applications can be run using this conduit and then instantiating a new process per sample evaluation (see: :ref:`Concurrent Execution Example <feature_concurrent.execution>`). 

For more information, see :ref:`Parallel Execution <parallel-execution>`. 

