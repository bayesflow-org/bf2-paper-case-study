#!/usr/bin/env python3
"""
This script presents the software replication material for the JSS submission titled
'BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python'

The code was tested using BayesFlow v2.0.10 (f9a7f2f)

To avoid excessive compute times, we recommend running this script with a CUDA-enabled backend.

Figures are generated in the `figures` directory.
A log file of all output is further saved as 'case-study.out'.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from contextlib import contextmanager
from time import perf_counter
from pathlib import Path
from pprint import pformat
from scipy.integrate import odeint
from scipy.stats import norm


@contextmanager
def timer(msg, suffix="..."):
    """A simple timer context manager that displays the time taken for wrapped code."""
    logging.info(f"{msg}{suffix}")
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        seconds = end - start
        logging.info(f"Done - took {seconds * 1e3:.0f} ms")


def trajectory_aggregation(traj, confidence=0.95):
    """Compute median and confidence bands for a collection of trajectories.

    Parameters
    ----------
    traj : np.ndarray
        Array of trajectories with shape (num_trajectories, num_timepoints)
    confidence : float
        Confidence level (default: 0.95 for 95% confidence bands)

    Returns
    -------
    central : np.ndarray
        Median trajectory
    L : np.ndarray
        Lower confidence band
    U : np.ndarray
        Upper confidence band
    """
    alpha = 1 - confidence
    quantiles = np.quantile(traj, [alpha / 2, 0.5, 1 - alpha / 2], axis=0).T
    central = quantiles[:, 1]
    L = quantiles[:, 0]
    U = quantiles[:, 2]
    return central, L, U


def plot_trajectories(
    samples,
    variable_keys,
    variable_names,
    fill_colors=("blue", "darkred"),
    legend_ncol=3,
    confidence=0.95,
    observations=None,
    ax=None,
    figsize=(11, 5),
):
    """Plot trajectories with confidence bands and optional observations.

    This function visualizes sampled trajectories by plotting the median trajectory,
    confidence bands, and individual trajectory samples. Optionally overlays
    observations.

    Parameters
    ----------
    samples : dict
        Dictionary containing sampled trajectories, must include key "t" for time vector
    variable_keys : list
        Keys to plot from samples dictionary
    variable_names : list
        Display names for variables
    fill_colors : tuple
        RGB color names for each variable
    legend_ncol : int
        Number of legend columns
    confidence : float
        Confidence level for bands (default: 0.95)
    observations : dict, optional
        Dictionary with observed data keyed as "observed_{key}" and "observed_t"
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on; if None, creates new figure
    figsize : tuple
        Figure size if creating new axes

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    t_span = samples["t"][0]

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        sns.despine()
    else:
        fig = ax.get_figure()

    for i, key in enumerate(variable_keys):
        if observations is not None:
            ax.scatter(
                observations["observed_t"],
                observations["observed_" + key],
                color=fill_colors[i],
                marker="x",
                label="Observed " + variable_names[i].lower(),
            )
            ax.axvline(
                observations["observed_t"].max(), color="black", linestyle="dashed"
            )

        central, L, U = trajectory_aggregation(samples[key], confidence=confidence)
        ax.plot(
            t_span,
            central,
            color=fill_colors[i],
            label="Median " + variable_names[i].lower(),
        )
        ax.fill_between(
            t_span,
            L,
            U,
            color=fill_colors[i],
            alpha=0.2,
            label=rf"{int((confidence) * 100)}$\%$ Confidence Bands",
        )

        # plot 20 trajectory samples to visualize posterior uncertainty
        for j in range(20):
            if j == 0:
                label = f"{variable_names[i]} trajectories"
            else:
                label = None
            ax.plot(
                t_span, samples[key][j], color=fill_colors[i], alpha=0.2, label=label
            )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=legend_ncol,
        frameon=True,
        fontsize=14,
    )
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel("population", fontsize=14)
    plt.tight_layout()

    return fig


def prior():
    """Sample prior parameters for the Lotka-Volterra model.

    The prior uses a logit-normal transformation to ensure parameters are in [0.1, 4].
    This transformation maps standard normal random variables to the desired range.

    Returns
    -------
    dict
        Dictionary with sampled prior parameters: alpha, beta, gamma, delta
    """
    x = np.random.normal(size=4)

    # logit normal distribution scaled to range from 0.1 and 4
    theta = 1 / (1 + np.exp(-x)) * 3.9 + 0.1

    return dict(
        alpha=theta[0],
        beta=theta[1],
        gamma=theta[2],
        delta=theta[3],
    )


def lotka_volterra_equations(state, t, alpha, beta, gamma, delta):
    """Lotka-Volterra predator-prey differential equations.

    The classic model for population dynamics:
    dx/dt = alpha*x - beta*x*y
    dy/dt = -gamma*y + delta*x*y

    Parameters
    ----------
    state : tuple
        Current populations (x, y) where x is prey and y is predator
    t : float
        Time (parameter for odeint, not used directly)
    alpha : float
        Prey growth rate
    beta : float
        Predation rate (prey death due to predators)
    gamma : float
        Predator mortality rate
    delta : float
        Predator efficiency (reproduction from prey consumption)

    Returns
    -------
    tuple
        Time derivatives (dx/dt, dy/dt)
    """
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y
    return dxdt, dydt


def ecology_model(
    alpha, beta, gamma, delta, t_span=(0, 5), t_steps=100, initial_state=(1, 1)
):
    """Simulate Lotka-Volterra predator-prey dynamics.

    Solves the Lotka-Volterra ODEs over the specified time span with given parameters.

    Parameters
    ----------
    alpha : float
        Prey growth rate parameter
    beta : float
        Predation rate parameter
    gamma : float
        Predator mortality parameter
    delta : float
        Predator efficiency parameter
    t_span : tuple
        Time interval (t_start, t_end)
    t_steps : int
        Number of time steps for the solution
    initial_state : tuple
        Initial populations (x0, y0)

    Returns
    -------
    dict
        Dictionary with keys 'x' (prey), 'y' (predator), and 't' (time vector)
    """
    t = np.linspace(t_span[0], t_span[1], t_steps)
    state = odeint(
        lotka_volterra_equations, initial_state, t, args=(alpha, beta, gamma, delta)
    )
    x, y = np.transpose(state, axes=(-1, -2))

    return dict(
        x=x,  # Prey time series
        y=y,  # Predator time series
        t=t,  # Time vector
    )


def observation_model(x, y, t, subsample=10, obs_prob=1.0, noise_scale=0.1):
    """Simulate noisy observations from ecological time series.

    Adds Gaussian noise to the simulated populations and subsamples time points
    to create realistic observation data.

    Parameters
    ----------
    x : np.ndarray
        Prey time series
    y : np.ndarray
        Predator time series
    t : np.ndarray
        Time vector
    subsample : int
        Interval for potential observation times (every subsample-th time point)
    obs_prob : float
        Probability of observing each subsampled time point
    noise_scale : float
        Standard deviation of Gaussian noise to add to observations

    Returns
    -------
    dict
        Dictionary with keys 'observed_x', 'observed_y', 'observed_t' containing
        noisy subsampled observations
    """
    t_steps = x.shape[0]

    # Add Gaussian noise to observations
    noisy_x = np.random.normal(x, noise_scale)
    noisy_y = np.random.normal(y, noise_scale)

    # Determine which time steps are observed
    step_indices = np.arange(0, t_steps, subsample)
    num_observed = int(obs_prob * len(step_indices))
    observed_indices = np.sort(
        np.random.choice(step_indices, size=num_observed, replace=False)
    )

    return {
        "observed_x": noisy_x[observed_indices],
        "observed_y": noisy_y[observed_indices],
        "observed_t": t[observed_indices],
    }


def period(observed_x, t_span=(0, 5), t_steps=500):
    """Compute the dominant period of a time series using a periodogram.

    Uses spectral analysis to identify the primary oscillation frequency in the data.

    Parameters
    ----------
    observed_x : np.ndarray
        Observed time series data
    t_span : tuple
        Time interval (t_start, t_end) matching the data
    t_steps : int
        Total number of time steps in the original data

    Returns
    -------
    float
        The dominant period (reciprocal of dominant frequency)
    """
    from scipy.signal import periodogram

    f, Pxx = periodogram(observed_x, t_steps / (t_span[1] - t_span[0]))
    freq_dominant = f[np.argmax(Pxx)]
    T = 1 / freq_dominant
    return T


def autocorr(trajectory, lags):
    """Compute autocorrelation for each specified lag in a time series.

    Measures how correlated a time series is with itself at different time lags,
    useful for understanding temporal dependencies in the data.

    Parameters
    ----------
    trajectory : np.ndarray
        The time series data, assumed to be a 1D array.
    lags : np.ndarray or list
        The lags at which to compute the autocorrelation.

    Returns
    -------
    auto_correlation : np.ndarray
        Autocorrelation values at each specified lag.
    """
    # Calculate the mean and variance of the trajectory for normalization
    mean = np.mean(trajectory)
    var = np.var(trajectory)

    # Initialize an array to hold the autocorrelation values
    auto_correlation = np.zeros(len(lags))

    # Compute autocorrelation for each lag
    for i, lag in enumerate(lags):
        if lag == 0:
            # Autocorrelation at lag 0 is always 1
            auto_correlation[i] = 1
        elif lag >= len(trajectory):
            # If the lag is equal to or greater than the length of the trajectory,
            # autocorrelation is undefined (set to 0)
            auto_correlation[i] = 0
        else:
            # Compute covariance and then autocorrelation
            cov = np.mean((trajectory[:-lag] - mean) * (trajectory[lag:] - mean))
            auto_correlation[i] = cov / var

    if np.any(np.isnan(auto_correlation)):
        print(
            f"Warning: NaN values found in autocorrelation: {auto_correlation}",
            file=sys.stderr,
        )

    return auto_correlation


def crosscorr(x, y):
    """Compute cross-correlation between two time series at zero lag.

    Computes the Pearson correlation coefficient between two aligned time series.

    Measures the linear association between two aligned time series.

    Parameters
    ----------
    x : np.ndarray
        The first time series data, assumed to be a 1D array of length n.
    y : np.ndarray
        The second time series data, assumed to be a 1D array of length n.

    Returns
    -------
    float
        The cross-correlation coefficient (ranges from -1 to 1).
    """
    # Compute the mean and standard deviation of both time series
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)

    # Compute the covariance and the correlation coefficient
    covariance = np.mean((x - mean_x) * (y - mean_y))
    correlation = covariance / (std_x * std_y)

    return correlation


def expert_stats(observed_x, observed_y, lags=(2, 5)):
    """Compute expert-defined summary statistics for ecological observations.

    Computes a fixed set of summary statistics from observed population trajectories.
    These statistics capture the key temporal and distributional features of the data
    and can be used for amortized inference without raw time series.

    Parameters
    ----------
    observed_x : np.ndarray with shape (num_observations, )
        Observed prey population data
    observed_y : np.ndarray with shape (num_observations, )
        Observed predator population data
    lags : tuple or list
        The lags at which to compute autocorrelation (default: (2, 5)).

    Returns
    -------
    dict with keys:
        means : np.ndarray with shape (2,)
            Mean of prey and predator populations
        log_vars : np.ndarray with shape (2,)
            Log-variance of prey and predator populations
        auto_corrs : np.ndarray with shape (2*len(lags),)
            Autocorrelations of both populations at specified lags
        cross_corr : float
            Correlation between prey and predator populations
        period : float
            Dominant oscillation period of prey population
    """
    means = np.array([observed_x.mean(), observed_y.mean()])
    log_vars = np.log(np.array([observed_x.var(), observed_y.var()]))
    auto_corrs = np.array(
        [
            autocorr(observed_x, lags),  # type: ignore
            autocorr(observed_y, lags),  # type: ignore
        ]
    ).flatten()
    cross_corr = crosscorr(observed_x, observed_y)
    T = period(observed_x)

    return dict(
        means=means,
        log_vars=log_vars,
        auto_corrs=auto_corrs,
        cross_corr=cross_corr,
        period=T,
    )


def take_posterior_sample(post_draws, dataset_id, sample_id):
    """Extract a single posterior sample for a specific dataset.

    Parameters
    ----------
    post_draws : dict
        Dictionary of posterior draws with shape (num_datasets, num_samples, ...)
    dataset_id : int
        Index of the dataset to extract
    sample_id : int
        Index of the posterior sample to extract

    Returns
    -------
    dict
        Dictionary with the same keys as post_draws, containing only the specified
        sample
    """
    posterior_sample_for_id = {
        var_key: post_draws[var_key][dataset_id, sample_id, ...].squeeze()
        for var_key in post_draws.keys()
    }
    return posterior_sample_for_id


def take_dataset(sims, dataset_id):
    """Extract a single dataset from a batch of simulations.

    Parameters
    ----------
    sims : dict
        Dictionary of simulations with first dimension indexing datasets
    dataset_id : int
        Index of the dataset to extract

    Returns
    -------
    dict
        Dictionary with the same keys as sims, containing only the specified dataset
    """
    return {var_key: sims[var_key][dataset_id] for var_key in sims.keys()}


def main():
    """Execute the complete case study as showcased in the paper.

    This script demonstrates BayesFlow 2's capabilities through a Lotka-Volterra
    predator-prey modeling example. It covers:

    1. Backend selection and determinism configuration
    2. Definition of observation models for simulating ecological data
    3. Building and sampling from simulators
    4. End-to-end posterior inference using raw time series data
    5. Expert summary statistics and point estimation approaches

    Resulting plots are saved to the 'figures' directory.
    """
    print(__doc__)

    figures_path = Path("figures")
    figures_path.mkdir(parents=True, exist_ok=True)

    np.set_printoptions(precision=2, threshold=10, edgeitems=2, suppress=True)

    # set up logging to both stdout and a log file
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("case-study.out"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Section 4.1: Preliminary: Choosing a Backend")

    import os

    # set the backend
    with timer("Setting the backend"):
        backend = "jax"
        os.environ["KERAS_BACKEND"] = backend

    # enable determinism individually for each backend
    with timer("Enabling determinism - note that this may negatively affect performance!"):
        seed = 2026
        match backend:
            case "jax":
                # as per https://github.com/jax-ml/jax/discussions/10674#discussioncomment-7214817
                os.environ["XLA_FLAGS"] = (
                    "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
                )
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
                os.environ["TF_CUDDN_DETERMINISTIC"] = "1"
            case "tensorflow":
                import tensorflow as tf  # type: ignore

                tf.config.experimental.enable_op_determinism()
            case "torch":
                import torch  # type: ignore

                # as per https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            case _:
                raise RuntimeError(
                    f"Cannot enable determinism for unknown backend: {backend!r}"
                )

    logging.info("Setting seed...")

    import keras

    keras.utils.set_random_seed(seed)

    logging.info("Section 4.2: Observation Model Definition")

    import bayesflow as bf

    logging.info(
        "We can easily inspect the sampled priors and the simulated observables with:\n"
        "observation_model(**ecology_model(**prior()))"
    )
    # Example output showing the structure of simulated observations
    simulation_output = observation_model(**ecology_model(**prior()))
    logging.info(pformat(simulation_output, indent=4))

    logging.info("Section 4.3: Simulator")

    with timer("Creating a simulator using bf.make_simulator"):
        # Chain together (prior -> ecology model -> observation model) into a Simulator object
        simulator = bf.make_simulator([prior, ecology_model, observation_model])

    num_trajectories = 100
    with timer(f"Sampling {num_trajectories} trajectories from the simulator"):
        samples = simulator.sample(num_trajectories)
        # Display shapes of the sampled data
        keras.tree.map_structure(keras.ops.shape, samples)

    logging.info(
        "We can inspect the sampled trajectories with:\n"
        "samples['alpha']\n" + pformat(samples["alpha"], indent=4) + "\n"
        "Or use them for diagnostics, or plotting:"
    )

    path = figures_path / "trajectory.pdf"
    with timer(f"Plotting an example trajectory at {path}"):
        fig = plot_trajectories(samples, ["x", "y"], ["Prey", "Predator"])
        fig.savefig(path)

    logging.info("Section 4.4: End-to-End Posterior Estimation With Raw Data")

    # Build adapter to transform raw simulation outputs into a network-compatible format
    with timer("Building the adapter for raw data"):
        adapter = (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .drop(["x", "y", "t"])  # Remove unobserved full trajectories
            .as_time_series(["observed_x", "observed_y", "observed_t"])
            .concatenate(
                ["alpha", "beta", "gamma", "delta"], into="inference_variables"
            )
            .concatenate(
                ["observed_x", "observed_y", "observed_t"], into="summary_variables"
            )
        )

    with timer("Creating neural networks"):
        # The summary network processes time series observations
        time_series_network = bf.networks.TimeSeriesNetwork(summary_dim=32)
        # We use flow matching as an inference network to learn the posterior p(θ|x).
        flow_matching = bf.networks.FlowMatching()

    with timer("Building workflow"):
        # Combine simulator, adapter, and networks into a workflow, used for easy fitting later.
        workflow = bf.BasicWorkflow(
            simulator=simulator,
            adapter=adapter,
            summary_network=time_series_network,
            inference_network=flow_matching,
        )

    with timer("Simulating training and validation data"):
        # Simulate large dataset for offline training
        training_data = workflow.simulate((5000,))
        # Simulate smaller dataset for validation during training
        validation_data = workflow.simulate((300,))

    with timer("Fitting workflow on pre-simulated data"):
        # Train the inference network
        workflow.fit_offline(
            data=training_data,
            epochs=100,
            batch_size=32,
            validation_data=validation_data,
        )

    with timer(f"Plotting default diagnostics at {figures_path}/"):
        fig_size = (20, 5)
        variable_names_latex = [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"]
        figs = workflow.plot_default_diagnostics(
            test_data=validation_data,
            variable_names=variable_names_latex,
            # The loss plot shows the convergence with respect to the training and validation set. Ideally, the training
            # and validation loss smoothly converge to a minimum. If the validation loss starts to increase again or is
            # much larger than the training loss, this indicates overfitting. Large spikes in the training loss can
            # indicate divergence, usually countered by clipping the loss gradient. Gradient clipping is enabled by
            # default in BayesFlow.
            loss_kwargs={
                "figsize": fig_size,
                "label_fontsize": 16,
                "title_fontsize": 20,
            },
            # The recovery plot shows how well the point estimates (mean and quantiles) match the true parameters
            # (targets). Ideally, samples lie on the diagonal (i.e., the estimates are close to the targets).
            recovery_kwargs={
                "figsize": fig_size,
                "label_fontsize": 16,
                "title_fontsize": 20,
            },
            # The calibration ECDF difference plot shows if the variance of the posterior is balanced (i.e., neither
            # under- nor overconfident). The model is well-calibrated if the resulting line plot (blue) is within the
            # selected quantile (gray shading).
            calibration_ecdf_kwargs={
                "figsize": fig_size,
                "legend_fontsize": 12,
                "difference": True,
                "label_fontsize": 16,
            },
            # The z-score contraction plot allows checking model sensitivity by showing the z-score of the true
            # parameters under the approximate posterior as a function of the contraction from the prior to the
            # posterior. Ideally, most samples lie in the high-contraction regime, spread evenly around a z-score of
            # zero.
            z_score_contraction_kwargs={"figsize": fig_size, "label_fontsize": 16},
        )

        for fig_name, fig in figs.items():
            fig.savefig(figures_path / ("e2e_" + fig_name + ".pdf"))

    with timer("Computing posterior samples"):
        # Sample from the trained amortized posterior for later use in plots
        estimates = workflow.sample(num_samples=300, conditions=validation_data)

    # pick a dataset that does not have significant parameter correlation
    dataset_id = 2

    path = figures_path / "e2e_posterior_pairs.pdf"
    with timer(f"Creating posterior pair plot at {path}"):
        # This pair plot shows the contraction of the posterior from the prior to the posterior distribution
        # for each parameter and pairwise combinations.
        # The blue points show the posterior samples, while the gray points show the prior samples.
        # Ideally, the posterior samples (blue) should be much more closely clustered around
        # the true parameter value (red x) than the prior samples (gray), indicating that the inference network
        # can extract significant information about the parameter from the observables.
        f = bf.diagnostics.plots.pairs_posterior(
            estimates=estimates,
            targets=validation_data,
            dataset_id=dataset_id,
            priors=validation_data,
            variable_names=variable_names_latex,
            label_fontsize=16,
            legend_fontsize=16,
        )
        f.savefig(path)

    with timer("Generating resimulations from posterior"):
        # Use posterior samples to resimulate extended trajectories
        num_post_samples = 300
        list_of_resimulations = []

        posterior_draws = workflow.sample(
            num_samples=num_post_samples, conditions=validation_data
        )

        # For each posterior sample, resimulate ecology model over extended time period
        for sample_id in range(num_post_samples):
            one_post_sample = take_posterior_sample(
                posterior_draws, dataset_id, sample_id
            )
            list_of_resimulations.append(
                ecology_model(t_span=(0, 20), **one_post_sample)
            )

        resimulation_samples = bf.utils.tree_stack(list_of_resimulations, axis=0)
        observations = take_dataset(validation_data, dataset_id)

    path = figures_path / "e2e_trajectories.pdf"
    with timer(f"Creating E2E trajectories at {path}"):
        f = plot_trajectories(
            resimulation_samples,
            ["x", "y"],
            ["Prey", "Predator"],
            observations=observations,
            legend_ncol=4,
            figsize=(12.5, 5),
        )
        f.savefig(path)

    # Point estimation has lower inferential power but significantly speeds up computation.
    logging.info("Section 4.5: Expert Summary Statistics and Point Estimation")

    with timer("Creating expert simulator with summary statistics"):
        # Build simulator that computes expert summary statistics instead of raw data
        expert_simulator = bf.make_simulator(
            [prior, ecology_model, observation_model, expert_stats]
        )

    with timer("Building expert adapter"):
        # Adapter for point estimation: drop raw data, keep only summary statistics
        expert_adapter = (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .drop(["x", "y", "t", "observed_x", "observed_y", "observed_t"])
            .concatenate(
                ["alpha", "beta", "gamma", "delta"], into="inference_variables"
            )
            .concatenate(
                ["means", "log_vars", "auto_corrs", "cross_corr", "period"],
                into="inference_conditions",
            )
        )

    quantile_levels = np.linspace(0.1, 0.9, 5)

    with timer("Creating point inference network"):
        # We use a scoring rule network to get point estimates of the mean and quantiles
        inference_network = bf.networks.ScoringRuleNetwork(
            scoring_rules=dict(
                mean=bf.scoring_rules.MeanScore(),
                quantiles=bf.scoring_rules.QuantileScore(quantile_levels),
            )
        )

    with timer("Building point estimation workflow"):
        # Workflow using summary statistics instead of raw time series
        expert_workflow = bf.BasicWorkflow(
            simulator=expert_simulator,
            adapter=expert_adapter,
            inference_network=inference_network,  # type: ignore
        )

    with timer("Training point estimation workflow (online)"):
        # Online training: sample on-the-fly during training rather than pre-simulating
        expert_workflow.fit_online(
            epochs=50,
            num_batches_per_epoch=200,
            batch_size=32,
        )

    with timer("Computing point estimates"):
        # Estimate posterior mean and quantiles
        expert_validation_data = expert_simulator.sample(300)
        point_estimates = expert_workflow.estimate(conditions=expert_validation_data)

    path = figures_path / "point_estimation_recovery.pdf"
    with timer(f"Creating recovery plot at {path}"):
        # Produce another recovery plot for the point estimates.
        marker_mapping = dict(quantiles="_", mean="*")
        f = bf.diagnostics.plots.recovery_from_estimates(
            estimates=point_estimates,  # type: ignore
            targets=expert_validation_data,
            variable_names=variable_names_latex,
            marker_mapping=marker_mapping,
        )
        f.savefig(path)

    path = figures_path / "point_estimation_calibration_ecdf.pdf"
    with timer(f"Creating calibration ECDF plot at {path}"):
        # Produce another calibration plot for the point estimates.
        f = bf.diagnostics.plots.calibration_ecdf_from_quantiles(
            estimates=point_estimates,  # type: ignore
            targets=expert_validation_data,
            quantile_levels=quantile_levels,
            difference=False,
            variable_names=variable_names_latex,
        )
        f.savefig(path)

    with timer("Generating posterior samples from point estimation quantiles"):
        # Convert quantile estimates to posterior samples via normal approximation
        std_normal_quantiles = norm.ppf(quantile_levels[[0, -1]])
        alpha_q = std_normal_quantiles[0] / (
            std_normal_quantiles[0] - std_normal_quantiles[1]
        )
        beta_q = (std_normal_quantiles[0] - 1) / (
            std_normal_quantiles[0] - std_normal_quantiles[1]
        )

        posterior_bounds_from_quantiles = keras.tree.map_structure(
            lambda v: v[:, [0, -1]],
            {k: v["quantiles"] for k, v in point_estimates.items()},
        )

        num_post_samples = 300

        # Generate posterior samples from the estimated quantiles
        posterior_draws_from_quantiles = keras.tree.map_structure(
            lambda v: np.random.normal(
                loc=v[:, 0] * (1 - alpha_q) + v[:, 1] * alpha_q,
                scale=v[:, 0] * (alpha_q - beta_q) + v[:, 1] * (beta_q - alpha_q),
                size=(300, num_post_samples),
            )[..., None],
            posterior_bounds_from_quantiles,
        )

    with timer("Generating resimulations from quantile-based posteriors"):
        # Resimulate trajectories from point estimation posterior samples
        dataset_id = 2
        list_of_resimulations = []

        for sample_id in range(num_post_samples):
            one_post_sample = take_posterior_sample(
                posterior_draws_from_quantiles, dataset_id, sample_id
            )
            list_of_resimulations.append(
                ecology_model(t_span=(0, 20), **one_post_sample)
            )

        resimulation_samples = bf.utils.tree_stack(list_of_resimulations, axis=0)
        observations = take_dataset(expert_validation_data, dataset_id)

    path = figures_path / "point_estimation_trajectories.pdf"
    with timer(f"Plotting point-estimated trajectories at {path}"):
        f = plot_trajectories(
            resimulation_samples,
            ["x", "y"],
            ["Prey", "Predator"],
            observations=observations,
            legend_ncol=4,
            figsize=(12.5, 5),
        )
        f.savefig(path)

    logging.info("Analysis complete!")


if __name__ == "__main__":
    main()
