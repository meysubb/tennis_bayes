import matplotlib.pyplot as plt
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.optim import Adam

def model_prep_svi(model_fn, iterations, lr, guide_fn: None):
    if guide_fn is None:
        guide = autoguide.AutoNormal(model_fn)
    else:
        guide = guide_fn(model_fn)

    optim = Adam(lr)
    svi = SVI(model_fn, guide, optim, loss=Trace_ELBO())
    return(guide, svi)

def svi_conv_plot(svi_res, model_name: str):
    loss = svi_res.losses
    fig, ax = plt.subplots()
    ax.plot(loss)
    ax.set_title(f'SVI Convergence for {model_name} Model')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO Loss')
    plt.close(fig)
    return(fig)