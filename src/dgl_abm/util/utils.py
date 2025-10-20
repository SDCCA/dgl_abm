"""This module contains miscelleneous utility functions.

Functions:
- sample_distribution_tensor: Acquires samples from different distributions
"""

import math
import torch


def sample_distribution_tensor(
    distribution_type: str,
    dist_parameters: list,
    samples: int | list | tuple,
    rounding: bool = False,
    decimals: None | int = None,
) -> torch.Tensor:
    """Generate and return samples from different distributions.

    Args:
        distribution_type (str): Type of distribution to sample
        dist_parameters (list): array of parameters as required/supported by
            requested distribution type
        samples (int|list|tuple): number or shape of samples to return
        rounding (bool): optional, whether the samples are to be rounded
        decimals (int): optional, required if rounding is specified; number of
            decimal places to round to
    Returns:
        torch.Tensor: samples from the specified distribution
    """
    sample_shape, dist_parameters = _format_inputs(samples, dist_parameters)

    dispatch = {
        "uniform": _uniform,
        "normal": _normal,
        "bernoulli": _bernoulli,
        "multinomial": _multinomial,
        "truncnorm": _truncnorm,
        "beta": _beta,
        "random": _random,
        "degenerate": _constant,
        "constant": _constant,
    }

    if distribution_type not in dispatch:
        message = (
            "Currently only uniform, normal, multinomial, beta, random, truncated normal"
            " bernoulli, and degenerate/constant distributions are supported."
        )
        raise NotImplementedError(message)

    dist = dispatch[distribution_type](dist_parameters, sample_shape)

    if rounding:
        if decimals is None:
            message = "Rounding functionality requires decimals of rounding accuracy to be specified."
            raise ValueError(message)
        return torch.round(dist, decimals=decimals)
    return dist


def _format_inputs(samples: int | list | tuple, dist_parameters: list) -> tuple[tuple, list]:
    """Format sample_shape as tuple and dist_parameters as torch tensors."""
    if not isinstance(samples, int | list | tuple):
        type_message = f"Type of samples (currently {type(samples)}) must be int, list, or tuple."
        raise TypeError(type_message)
    if isinstance(samples, int):
        sample_shape = (samples,)
    elif isinstance(samples, list):
        sample_shape = tuple(samples)
    elif isinstance(samples, tuple):
        sample_shape = samples

    for i, item in enumerate(dist_parameters):
        if item is not None and not isinstance(item, torch.Tensor):
            dist_parameters[i] = torch.tensor(item)
    return sample_shape, dist_parameters


def _uniform(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a uniform distribution (effectively the same as "random" type).

    Note: dist_parameters[0] is min
          dist_parameters[1] is max
    """
    if dist_parameters[0] == dist_parameters[1]:
        message = (
            "Uniform distribution requires first parameter (min) to be less "
            "than second parameter (max). Use a degenerate/constant distribution if "
            "the desired effect is a sample of constant value."
        )
        raise ValueError(message)
    return torch.distributions.uniform.Uniform(
        dist_parameters[0].to(torch.float32), dist_parameters[1].to(torch.float32)
    ).sample(sample_shape)


def _normal(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a normal distribution.

    Note: dist_parameters[0] is mean
          dist_parameters[1] is standard deviation
    """
    return torch.distributions.normal.Normal(
        dist_parameters[0].to(torch.float32), dist_parameters[1].to(torch.float32)
    ).sample(sample_shape)


def _bernoulli(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a bernoulli distribution.

    Note: dist_parameters[0] is probs
          dist_parameters[1] is logits
    """
    return torch.distributions.bernoulli.Bernoulli(
        probs=dist_parameters[0], logits=dist_parameters[1], validate_args=None
    ).sample(sample_shape)


def _multinomial(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a multinomial distribution.

    Note: dist_parameters[0] is the tensor of probabilities
          dist_parameters[1] is the tensor of possible outcomes
    """
    n_samples = sample_shape[0] if len(sample_shape) == 1 else math.prod(sample_shape)
    multinomial_samples = torch.multinomial(
        torch.tensor(dist_parameters[0].to(torch.float32)), n_samples, replacement=True
    )
    dist = torch.gather(torch.Tensor(dist_parameters[1]), 0, multinomial_samples)
    if len(sample_shape) > 1:
        dist = dist.reshape(sample_shape)
    return dist


def _truncnorm(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a truncated normal distribution.

    Note: dist_parameters[0] is mean
          dist_parameters[1] is standard deviation
          dist_parameters[2] is min
          dist_parameters[3] is max
          Mathematical basis for the truncated normal sampling code is
          cdf(x)=(1+erf(x/2^0.5))/2 and inverse cdf^-1(x)=2^0.5*erfinv(2*x-1).
    """
    trunc_val_min = (dist_parameters[2] - dist_parameters[0]) / dist_parameters[1]
    trunc_val_max = (dist_parameters[3] - dist_parameters[0]) / dist_parameters[1]
    cdf_min = (1 + torch.erf(trunc_val_min / torch.sqrt(torch.tensor(2.0)))) / 2
    cdf_max = (1 + torch.erf(trunc_val_max / torch.sqrt(torch.tensor(2.0)))) / 2

    uniform_samples = torch.rand(sample_shape)
    inverse_transform = torch.erfinv(2 * (cdf_min + (cdf_max - cdf_min) * uniform_samples) - 1)
    sample_ppf = torch.sqrt(torch.tensor(2.0)) * inverse_transform

    return dist_parameters[0] + dist_parameters[1] * sample_ppf


def _beta(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a beta distribution.

    Note: dist_parameters[0] is alpha
          dist_parameters[1] is beta
    """
    return torch.distributions.beta.Beta(dist_parameters[0], dist_parameters[1]).sample(sample_shape)


def _random(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a random distribution.

    Note: dist_parameters can be (a) empty, (b) dist_parameters[0] is maximum (with
          implicit 0 minimum), or (c) dist_parameters[0] is minimum and dist_parameters[1]
          is maximum (effectively the same as "uniform" type).
    """
    dist = torch.rand(sample_shape)
    if len(dist_parameters) == 0:
        pass
    elif len(dist_parameters) == 1:
        dist = dist * (dist_parameters[0])
    elif len(dist_parameters) == 2:  # noqa: PLR2004
        dist = dist * (dist_parameters[1] - dist_parameters[0]) + dist_parameters[0]
    else:
        message = "Random distribution supports 0, 1 (max), or 2 (min/max) parameters"
        raise ValueError(message)
    return dist


def _constant(dist_parameters: list, sample_shape: tuple) -> torch.Tensor:
    """Generate samples from a degenerate/constant distribution.

    Note: dist_parameters[0] is the constant value; if that value has multiple dimensions,
    the samples must be requested as an int or single-element tuple or list. I.e., that value
    will only be broadcast across a 1-D sample shape.
    """
    if len(dist_parameters) != 1:
        length_message = "Degenerate/constant distributions require exactly 1 parameter"
        raise ValueError(length_message)
    if dist_parameters[0].ndim > 1 or (dist_parameters[0].ndim == 1 and dist_parameters[0].shape[0] > 1):
        if len(sample_shape) == 1:
            dist = dist_parameters[0].unsqueeze(0).repeat(sample_shape[0], 1)
        else:
            message = (
                "If the constant value is multidimensional, samples must be provided ",
                "as an int or single-element tuple or list",
            )
            raise ValueError(message)
    else:
        dist = torch.full(sample_shape, dist_parameters[0].item())
    return dist
