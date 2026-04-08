*****************************
Bayesian Inference
*****************************

..
  In a *Bayesian Inference* problem, the [???] defines and a prior probability density $p(\vartheta)$ for the problem variables,
  and the solver is applied to the [???] of the posterior distribution:

In a *Bayesian Inference* problem, we have a probability model that consists of a conditional probability :math:`p(d | \vartheta)`
of data :math:`d` given a set of variables :math:`\vartheta`, and a prior :math:`p(\vartheta)` for the problem variables.

The solver is applied to the posterior distribution of the problem variables:

.. math::

   p(\vartheta | d) = \frac{p(d | \vartheta) p(\vartheta)}{p(d)}


Subtypes
~~~~~~~~~

For data stemming from a computational model, for which you only want to choose a noise distribution and variable prior,
a Bayesian *Reference Likelihood* problem can be used.

A *Custom Likelihood* problem allows any kind of user defined probability
model :math:`p(d | \vartheta)`.

If your problem has additional (latent) variables in addition to the parameters of interest, the subtypes of Bayesian
*Latent Variable* problems can be used.

.. Todo:
   Approximate
