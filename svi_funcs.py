import time

import pyro


def run_svi(model,
            model_args,
            guide,
            lr=0.05,
            betas=(0.9, 0.999),
            steps=250,
            seed=-1):
    """
    Run SVI with input parameters.
    """

    pyro.clear_param_store()

    if seed >= 0:
        pyro.set_rng_seed(seed)

    adam = pyro.optim.Adam({"lr": lr, "betas": betas})  # 0.005 gets approx. lowest ELBO loss
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, guide, adam, elbo)

    start = time.time()

    losses = {}
    for step in range(steps):
        losses[step] = svi.step(*model_args)
        if (step + 1) % 50 == 0:
            print(f"step: {step:>5}, ELBO loss: {losses[step]:.2f}")

    print(f"\nfinished in {time.time() - start:.2f} seconds")

    return losses, pyro.get_param_store().items()
