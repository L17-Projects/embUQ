**************************
Extending Korali
**************************

Korali can be extended with new algorithms and problems by adding modules. Modules are plug-and-play packages consisting of base C++ code (`.cpp.base`, `.hpp.base` files), configuration (`.config` file), and documentation (`README.rst` file).

To create a new Korali module, you need to create a new folder inside the `/source/modules/` folder. You will find examples of existing modules that you can use as basis for your new module. After creating the module, you need to add it to the list of selectable modules in the `source/modules/module.cpp` file.

Base Code
*************************

The base module code files (`.base`) contain the definition of the Module's C++ class and its component functions. The base files contain certain markers that trigger the automatic generation of code upon building. These markers should not be removed. Building Korali will result in end `.hpp` and `.cpp` files. These files should be included in the git repository as generated and never manually modified (any changes will be overwritten in the next build).

Documentation
*************************

In the `README.rst` file, the author of the module must add a detailed description of the module's purpose and rationale, including any publications that can serve as further explanation.

Configuration
**************************

After creating the new folder, you need to create a configuration file with extension `.config` that describes the internal aspects of the module. We explain below the purpose of each of the category contained therein.

Module Data
-------------------------

This category contains basic information about the Module (e.g., class name and namespace). This information should be accurately defined for the build to work. 

Configuration Settings
-------------------------

Each of the entries in this category represents a user-configurable aspect of the module. Upon building Korali, they become public fields in the module's C++ class. 

Variables Configuration
--------------------------------------

Each of the entries in this category represents a parameter of the Korali experiment's variables. These settings are user-configurable, and they are accessed by prefixing ["Variables"][numerical_index]. Upon building Korali, they become fields in the korali::Variable class file.

Internal Settings
--------------------------------------

Each of the entries in this category represents a developer-only aspect of the module. These settings are not user-configurable, but their purpose is to automatize the serialization/deserialization of the internal state of the module. Upon building Korali, they become fields in the module's C++ class file.

Module Defaults
--------------------------------------

Defines the default configuration of all the parameters that describe the module. They are overwritten by any user-specified values.

Variable Defaults
--------------------------------------

Defines the default configuration for parameters of the korali::Variable class. They are applied to all the variables defined in the experiment. They are overwritten by any user-specified values.  

(Solvers Only) Termination Criteria
--------------------------------------

Each of the entries in this category determines a criterion that Korali checks at each generation to determine whether to continue or finish execution. These settings are user-configurable, but they are accessed by prefixing ["Solver"]["Termination Criteria"]. Upon building Korali, they become fields in the module's C++ class with prefix _terminationCriteria.

(Problems Only) Compatible Solvers
--------------------------------------

List of all the solvers that can be used with the current problem type. Solvers are specified by name and must produce at least one result.  

(Problems Only)  Results
--------------------------------------

List of all the results that can be obtained by running this problem. Each result specifies which solvers can produce them. It is the task of the developer to make sure that the promised results are indeed produced by the solvers here specified.

(Problems Only) Available Operations
--------------------------------------

Lists all the operations that the problem can perform on a sample. Each operation links to an actual C++ method in the class.  
