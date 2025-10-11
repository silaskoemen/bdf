"""Classes for using parametric distributions for split finding (e.g., Normal, Exponential, etc.), and Kernel Density Estimation (KDE)
in the final leaves - Kernel Density Leaves (KDL).

Leverages already implemented BDFDistribution classes, routes for split finding and prediction.
"""


"""Thoughts on how to do this:
- `nll` should use _dist distribution for float return for split finding
- `predict` of Node uses saved params, so {'data': y, 'bandwidth': bw} for kde should be returned
- Conflict of internal `calc_posterior_params` needed for split finding, BUT KDE params needed for prediction
- Need info on whether split finding or prediction is being done during posterior param calculation
- Could also always calculate both & use _dist for `nll` but actually return kde params, but seems inefficient
"""
