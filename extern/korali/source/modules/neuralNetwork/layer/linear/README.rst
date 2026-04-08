*************
Linear Layer
*************

Specialization of the Layer for Linear Maps. The entries of the previous layer :math:`\mathbf{z}^{l-1}\in\mathbb{R}^{n_{l-1}}` are multiplied by a weight matrix :math:`W\in\mathbb{R}^{n_{l}\times n_{l-1}}` and a bias :math:`\mathbf{b}^l\in\mathbb{R}^{n_l}`. 

.. math::

    \mathbf{z}^l = W^l \mathbf{z}^{l-1}+\mathbf{b}^l

If we denote the components of the vectors by :math:`z_i^{l-1}` and :math:`z_j^{l}, b_j^l` and for the matrix by :math:`W_{ij}^{l}` for :math:`i=1,\dots,n_{l-1}` and :math:`j=1,\dots,n_{l}`, this operation can be written as

.. math::

    z_j^l = W_{ij}^l z_i^{l-1}+b_j^l
    
