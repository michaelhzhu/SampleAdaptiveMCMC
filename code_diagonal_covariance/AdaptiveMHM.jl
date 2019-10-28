module AdaptiveMHM


using Distributions
using AtomsM, Models
using MHM
export AdaptiveMH


function AdaptiveMH(atoms, kappa, K, p, q0, q, args)

    if atoms.num_atoms != 1; error(); end

    # Initialize the state
    N = atoms.num_atoms
    theta_set = rand(q0, N)
    for i in 1:N
        save_atom(atoms, theta_set[:,i], logpdf(p, theta_set[:,i]), i)
    end

    dim_theta = size(atoms.theta, 1)
    mu_k = zeros(dim_theta)
    S_k = zeros(dim_theta)

    # Run kappa burn-in iterations using fixed proposal distribution
    # Run K-kappa estimation iterations using adaptive proposal
    for k in 1:K
        MH_step(atoms, p, q, 1)
        snapshot(atoms)
        
        mu_k, S_k = update_mean_and_var(k, mu_k, S_k, atoms, args)
        sigma = sqrt.(S_k/(k-1))
        if k >= kappa
            # Update covariance matrix
            q = MvNormal(args["adaptive_scale"]*sigma)
        end

        if k % args["print_every"] == 0
            print(k, ": ", logpdf(p, vec(atoms.theta)), "\t")
            print_errors(p, atoms)
            println()
            flush(STDOUT)
        end
    end
end


# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
function update_mean_and_var(k, mu_prev, S_prev, atoms, args)
    x = atoms.theta[:, 1]
    mu = mu_prev + (x - mu_prev)/k
    S = S_prev + (x - mu_prev).*(x - mu)
    return mu, S
end


end
