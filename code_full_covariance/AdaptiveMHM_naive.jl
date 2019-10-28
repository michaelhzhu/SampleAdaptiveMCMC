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
    theta_history = zeros(dim_theta, 1, K)

    # Run kappa burn-in iterations using fixed proposal distribution
    # Run K-kappa estimation iterations using adaptive proposal
    for k in 1:K
        MH_step(atoms, p, q, 1)
        snapshot(atoms)

        theta_history[:, 1, k] = copy(atoms.theta[:, 1])
        if k >= kappa
            # Update covariance matrix
            @views cov_matrix = cov(theta_history[:, 1, 1:k], 2)
            q = MvNormal(args["adaptive_scale"]^2*cov_matrix)
        end

        if k % args["print_every"] == 0
            print(k, ": ", logpdf(p, vec(atoms.theta)), "\t")
            print_errors(p, atoms)
            println()
            flush(STDOUT)
        end
    end
end


end
