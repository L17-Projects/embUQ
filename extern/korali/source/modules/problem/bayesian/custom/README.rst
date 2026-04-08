**************************
Custom Likelihood
**************************

..
  In a *Bayesian Inference* problem, the [???] defines and a prior probability density $p(\vartheta)$ for the problem variables,
  and the solver is applied to the [???] of the posterior distribution:

..
  In a *Bayesian Inference* problem, we have a probability model that consists of a conditional probability :math:`p(d | \vartheta)`
  of data :math:`d` given a set of variables :math:`\vartheta`, and a prior :math:`p(\vartheta)` for the problem variables.

  The solver is applied to the posterior distribution of the problem variables:

  .. math::

     p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)}

..
  ## Commented-out end ##


While a Bayesian *Reference* type problem is for data that originate from a functional dependency,
:math:`d = (x_j, y_j)_{j=1...N}\;`  with :math:`y_j = f(x_j) + \epsilon`,
a *Custom Likelihood* model makes no such assumption.

With a *Custom Likelihood*, the function :math:`p(d|\vartheta)` is given directly by a user-defined model of the form :math:`f:\; \mathbb{R}^N\rightarrow\mathbb{R}`, where :math:`N` is the number of variables.

..
  Note: I didn't want to delete the next paragraph - can someone double check where it belongs? Is it
        part of the description of Reference?

Likelihood Models
-------------------

Additive Normal Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whereas with an *Additive Normal Likelihood*, the computational model is assumed to be of the form :math:`f(x;\vartheta)`, where :math:`d` is a set of *M* given data points. The output of the model represents the values of the function at the given points for which Korali can build a likelihood function :math:`p(d|\vartheta)`, and a prior probability density :math:`p(\vartheta)`.

Currently, Korali uses a Normal estimator for the error component of the likelihood calculation, using a statistical-type variable, *sigma*:

.. math::

   p(d | \vartheta) = {\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left((x-\mu )/\sigma \right)^{2}}

With a *Custom Likelihood*, the function :math:`p(d|\vartheta)` is given directly by a user-defined model of the form :math:`f:\mathbb{R}^N\rightarrow\mathbb{R}`, where :math:`N` is the number of variables.

