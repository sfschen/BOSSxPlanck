# Analytic marginalization.

The default Cobaya sampler scales reasonably poorly to a large number of parameters,
so significant gains in runtime can be achieved by means of reducing the number of
sampled parameters.  This is particularly easy for parameters that enter the model
prediction only linearly, since for Gaussian priors on those parameters the marginalization
can be performed analytically (by integrating a Gaussian!).

This set of codes implements the analytic marginalization of linear parameters for the
galaxy-cross-lensing likelihood.
