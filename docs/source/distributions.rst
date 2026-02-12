Distributions
=============

BDF supports 25+ distribution implementations. Each distribution is selected
via the ``dist`` parameter of :class:`~bdf.tree_classes.bdf_regressor.BDFRegressor`
or :class:`~bdf.tree_classes.bdf_regressor.BDFClassifier`, and configured via
the ``params`` dictionary.

Common Parameters
-----------------

All distributions inherit from :class:`~bdf.distributions.BDFDistributionParams`,
which provides shared scoring and inference options. Individual distributions
document only the parameters they add or restrict.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.BDFDistributionParams


Continuous Distributions
------------------------

Normal
~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.NormalMuNormal
   bdf.distributions.NormalMuInvGammaSigmaNormal

Student-t
~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.FrequentistStudentT
   bdf.distributions.NormalMeanStudentT

Skew-Normal
~~~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.NormalMeanPseudoAlphaSkewNormal
   bdf.distributions.NormalMeanNormalGammaSkewNormal

Generalized Hyperbolic
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.FrequentistGenHyperbolic


Count Distributions
-------------------

Poisson
~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.GammaABLambdaPoisson
   bdf.distributions.GammaMVLambdaPoisson

Exponential
~~~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.GammaABLambdaExponential
   bdf.distributions.GammaMVLambdaExponential

Negative Binomial
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.FrequentistNegativeBinomial
   bdf.distributions.NormalMeanNegativeBinomial
   bdf.distributions.GammaMSLambdaNegBin


Discrete / Classification
--------------------------

Bernoulli
~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.BetaABBernoulli
   bdf.distributions.BetaMVBernoulli

Multinomial
~~~~~~~~~~~
.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.DirichletAlphaMultinomial
   bdf.distributions.DirichletMeanMultinomial


Non-parametric
--------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   bdf.distributions.KDE
