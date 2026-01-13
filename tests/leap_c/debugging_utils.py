"""Utilities for debugging.

Not necessarily meant to be imported yourself,
but also as a code-block templates for functionality to be copy-pasted
and adjusted according to one's needs.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def summarize_gradients(a, n, rtol, atol):
    """Useful for debugging the sensitivities.

    I inserted it somewhere around "if not _allclose_with_type_promotion(a, n, rtol, atol):"
    in gradcheck.py when running the pytorch gradcheck tests.

    Args:
        a: Analytic gradient (IFT)
        n: Numerical gradient (FD)
        rtol: Relative tolerance for torch.isclose
        atol: Absolute tolerance for torch.isclose
    """
    print("max:", a.max(), n.max())
    print("min:", a.min(), n.min())
    print("Max difference:", (a - n).abs().max())
    neq = ~torch.isclose(a, n, rtol=rtol, atol=atol)
    nonzero = ~torch.isclose(a, torch.zeros_like(a), rtol=rtol, atol=atol)
    nnonzero = ~torch.isclose(a, torch.zeros_like(n), rtol=rtol, atol=atol)
    print("Wrong elements: ", neq.sum())
    print("Nonzero elements: ", (nonzero.sum()))
    print("Indices of wrong elements: ", torch.nonzero(neq, as_tuple=True))
    nonzero_ind_a = torch.nonzero(nonzero, as_tuple=True)
    nonzero_ind_n = torch.nonzero(nnonzero, as_tuple=True)
    for entry_a, entry_n in zip(nonzero_ind_a, nonzero_ind_n):
        if not torch.allclose(entry_a, entry_n):
            print("Indices of nonzero elements a: ", nonzero_ind_a)
            print("Indices of nonzero elements n: ", nonzero_ind_n)
            raise Exception("Nonzero indices do not match")
        else:
            print("Nonzero indices match")


def investigate_single_gradient_entry(diff_mpc, test_inputs, test_func_template):
    """Plots the sensitivities and function along a single dimension.

    Adjust this for plotting of single entries of partial derivatives. Insert, e.g.,
    in except-block of check_gradients when a specific gradient test fails (e.g., dQdx0).

    Args:
        diff_mpc: The differentiable mpc
        test_inputs: The inputs used for testing.
        test_func_template: The template function to create the test function
            (e.g., _create_dQdx0_test).
    """
    sample_idx = 1
    dim_idx = 1
    num = 100
    granularity = 1e-3
    sample_gradient_hardcoded = 1.8537e-02

    sample = test_inputs.x0[[sample_idx]].detach().cpu().numpy()

    samples = np.tile(sample, (num, 1))
    samples[:, dim_idx] = (
        granularity * np.linspace(0, 1, num=num) + sample[:, dim_idx] - 0.5 * granularity
    )
    samples = torch.tensor(samples)
    samples.requires_grad = True
    u0 = torch.tile(test_inputs.u0[[sample_idx]], (100, 1))
    test_func = test_func_template(diff_mpc, u0)

    fwd_out, status = test_func(samples)
    assert torch.all(status == 0)
    fwd_out.sum().backward()
    partial_der_entry = samples.grad[:, dim_idx]  # type:ignore

    # Create figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = samples[:, dim_idx].detach().numpy()
    ax1.plot(x, partial_der_entry.detach().numpy(), "b-", linewidth=2)
    ax1.scatter(sample[0, dim_idx], sample_gradient_hardcoded, s=2, c="green", marker="+")
    ax1.set_title("partial der")
    ax1.set_xlabel("x")
    ax1.set_ylabel("partial der")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, fwd_out.detach().numpy(), "r-", linewidth=2)
    ax2.set_title("Q-func")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Q-func")
    ax2.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig("inspect_derivatives")
