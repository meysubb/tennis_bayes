import numpyro 
import numpyro.distributions as dist

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

    # Surface 
    mu_surface = numpyro.sample("mu_surface", dist.Normal(0, 1))
    sigma_surface = numpyro.sample("sigma_surface", dist.HalfNormal(1))
    with numpyro.plate("surface", surface_n):
        surface_effect = numpyro.sample("surface_effect", dist.Normal(mu_surface, sigma_surface))

    # Combine logits
    logits = intercept + player_server[server_idx] - player_return[returner_idx] + surface_effect[surface_idx]

    # Likelihood
    numpyro.sample("obs", dist.Binomial(total_count=serve_attempts, logits=logits), obs=serve_pts_won)    