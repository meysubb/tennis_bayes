from matplotlib.pylab import gamma
import numpyro 
import numpyro.distributions as dist
from jax import numpy as jnp 
from jax import lax

def basic_serve_return_model(X_intercept, n_players, server_idx, returner_idx, serve_attempts, serve_pts_won):
    """
    Hierarchical model with separate serve and return effects for each player.
    Args:
        Intercept: prior mean for intercept
        players_n: number of players
        server_idx: array of server indices (0-based)
        returner_idx: array of returner indices (0-based)
        y: binary outcome (1=server wins point, 0=loses)
    """
    # Intercept
    intercept = numpyro.sample("intercept", dist.Normal(X_intercept, 1))

    # Random Effects
    # Server
    mu_server = numpyro.sample("mu_server", dist.Normal(0, 1))
    sigma_server = numpyro.sample("sigma_server", dist.HalfNormal(1))
    with numpyro.plate("players_server", n_players):
        player_server = numpyro.sample("player_server", dist.Normal(mu_server, sigma_server))


    # Returner
    mu_return = numpyro.sample("mu_ret", dist.Normal(0, 1))
    sigma_return = numpyro.sample("sigma_ret", dist.HalfNormal(1))
    with numpyro.plate("players_returner", n_players):
        player_return = numpyro.sample("player_ret", dist.Normal(mu_return, sigma_return))

    # Combine logits
    logits = intercept + player_server[server_idx] - player_return[returner_idx]

    # Likelihood
    numpyro.sample("obs", dist.Binomial(total_count=serve_attempts, logits=logits), obs=serve_pts_won)    


def serve_return_surface_model(X_intercept, n_players, 
                               server_idx, returner_idx, 
                               surface_idx, surface_n,
                               serve_attempts, serve_pts_won):
    """
    Hierarchical model with separate serve, return and surface effects for each player.
    Args:
        Intercept: prior mean for intercept
        players_n: number of players
        server_idx: array of server indices (0-based)
        returner_idx: array of returner indices (0-based)
        y: binary outcome (1=server wins point, 0=loses)
    """
    # Intercept
    intercept = numpyro.sample("intercept", dist.Normal(X_intercept, 1))

    # Random Effects
    # Server
    mu_server = numpyro.sample("mu_server", dist.Normal(0, 1))
    sigma_server = numpyro.sample("sigma_server", dist.HalfNormal(1))
    with numpyro.plate("players_server", n_players):
        player_server = numpyro.sample("player_server", dist.Normal(mu_server, sigma_server))

    # Returner
    mu_return = numpyro.sample("mu_ret", dist.Normal(0, 1))
    sigma_return = numpyro.sample("sigma_ret", dist.HalfNormal(1))
    with numpyro.plate("players_returner", n_players):
        player_return = numpyro.sample("player_ret", dist.Normal(mu_return, sigma_return))

    # Surface 
    mu_surface = numpyro.sample("mu_surface", dist.Normal(0, 1))
    sigma_surface = numpyro.sample("sigma_surface", dist.HalfNormal(1))
    with numpyro.plate("surface", surface_n):
        surface_effect = numpyro.sample("surface_effect", dist.Normal(mu_surface, sigma_surface))

    # Combine logits
    logits = intercept + player_server[server_idx] - player_return[returner_idx] + surface_effect[surface_idx]

    # Likelihood
    numpyro.sample("obs", dist.Binomial(total_count=serve_attempts, logits=logits), obs=serve_pts_won)    



def serve_return_surface_ar_model(X_intercept, n_players, 
                            server_idx, returner_idx, 
                            surface_idx, surface_n,
                            serve_year_idx, return_year_idx, 
                            max_years, 
                            serve_attempts, serve_pts_won):
    """
    Hierarchical model with separate serve, return, surface effects for each player.
    Separate Serve/Return effect by year using an AR(1) structure. 
    Args:
        Intercept: prior mean for intercept
        players_n: number of players
        server_idx: array of server indices (0-based)
        returner_idx: array of returner indices (0-based)
        y: binary outcome (1=server wins point, 0=loses)
    """
    # Intercept
    intercept = numpyro.sample("intercept", dist.Normal(X_intercept, 1))

    # Random Effects
    mu_server = numpyro.sample("mu_server", dist.Normal(0, 1))
    # AR effects - Serve
    gamma_serve = numpyro.sample("gamma_serve", dist.Beta(3, 1))
    # initial year variance
    sigma_serve_0 = numpyro.sample("sigma_serve_0", dist.HalfNormal(1))
    # future years variance
    sigma_serve_yr = numpyro.sample("sigma_serve_yr", dist.HalfNormal(1))

    with numpyro.plate("players_serve", n_players):
        player_serve = numpyro.sample("player_serve_init", dist.Normal(mu_server, sigma_serve_0))

    with numpyro.plate("players_serve_next", n_players):
        with numpyro.plate("years", max_years):
            player_serve_next = numpyro.sample("player_serve_next", 
                                               dist.Normal(mu_server, sigma_serve_yr))
            
    # general idea for AR(1) here 
    #https://stackoverflow.com/questions/78209454/numpyro-ar1-mean-switching-model-sampling-incongrouencies
    def make_ar_effect(gamma):
        def ar_effect(prev_state, update):
            next_state = gamma * prev_state + update
            return next_state, next_state
        return ar_effect
    
    #_, serve_effects_scan = lax.scan(ar_effect, player_serve, player_serve_next)
    _, serve_effects_scan = lax.scan(make_ar_effect(gamma_serve), player_serve, player_serve_next)
    serve_effects = jnp.concatenate([player_serve[:, None], serve_effects_scan.T], axis = 1)

    # Returner
    mu_return = numpyro.sample("mu_ret", dist.Normal(0, 1))
    # AR effects - Return
    gamma_return = numpyro.sample("gamma_return", dist.Beta(3, 1))
    # initial year variance
    sigma_return_0 = numpyro.sample("sigma_return_0", dist.HalfNormal(1))
    # future years variance
    sigma_return_yr = numpyro.sample("sigma_return_yr", dist.HalfNormal(1))
    
    with numpyro.plate("players_returner", n_players):
        player_return = numpyro.sample("player_ret", dist.Normal(mu_return, sigma_return_0))

    with numpyro.plate("players_return_next", n_players):
        with numpyro.plate("years", max_years):
            player_return_next = numpyro.sample("player_return_next", 
                                               dist.Normal(mu_return, sigma_return_yr))


    #_, return_effects_scan = lax.scan(ar_effect, player_return, player_return_next)
    _, return_effects_scan = lax.scan(make_ar_effect(gamma_return), player_return, player_return_next)
    return_effects = jnp.concatenate([player_return[:, None], return_effects_scan.T], axis = 1)

    # Surface 
    mu_surface = numpyro.sample("mu_surface", dist.Normal(0, 1))
    sigma_surface = numpyro.sample("sigma_surface", dist.HalfNormal(1))
    with numpyro.plate("surface", surface_n):
        surface_effect = numpyro.sample("surface_effect", dist.Normal(mu_surface, sigma_surface))

    # Combine logits
    serve_full_effect = serve_effects[server_idx, serve_year_idx]
    return_full_effect = return_effects[returner_idx, return_year_idx]
    
    logits = intercept + serve_full_effect - return_full_effect + surface_effect[surface_idx]

    # Likelihood
    numpyro.sample("obs", dist.Binomial(total_count=serve_attempts, logits=logits), obs=serve_pts_won)    