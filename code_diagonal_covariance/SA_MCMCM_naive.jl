module SA_MCMCM


using Distributions
using AtomsM, Models
export SA_MCMC


function SA_MCMC(atoms::Atoms, K::Int, p::ContinuousMultivariateDistribution, q0, args)
    
    # Initialize the state
    N = atoms.num_atoms
    theta_set = rand(q0, N)
    for i in 1:N
        save_atom(atoms, theta_set[:,i], logpdf(p, theta_set[:,i]), i)
    end
    
    for k=1:K
        theta_new = step_propose(atoms, p, args)
        step_substitute(atoms, theta_new, p, args)
        snapshot(atoms)

        if k % args["print_every"] == 0
            print(k, ": ", logpdf(p, vec(mean(atoms.theta, 2))), "\t")
            print(maximum(atoms.logp_theta), "\t")
            print(minimum(atoms.logp_theta), "\t")
            print_errors(p, atoms)
            println()
            flush(STDOUT)
        end
    end
end


function step_propose(atoms::Atoms, p::ContinuousMultivariateDistribution, args)
    mu = vec(mean(atoms.theta, 2))
    sigma = vec(std(atoms.theta, 2, corrected=false, mean=mu))
    
    q = construct_mixture_distribution(mu, sigma, args)
    theta_new = rand(q)
    return theta_new
end


function step_substitute(atoms::Atoms, theta_new::Array{Float64, 1}, p::ContinuousMultivariateDistribution, args)
    # Compute the probabilities of substituting the proposed theta_new into each index of atoms.theta
    # as well as rejecting theta_new
    lambda_n, logp_theta_new = get_substitution_probs(atoms, theta_new, p, args)

    # Substitute in the new atom (if not rejected)
    dist = Categorical(lambda_n/sum(lambda_n))
    index = rand(dist)
    if index < atoms.num_atoms+1
        save_atom(atoms, theta_new, logp_theta_new, index)
        atoms.last_acc = true
    else
        atoms.last_acc = false
    end
end


function get_substitution_probs(atoms::Atoms, theta_new::Array{Float64, 1}, p::ContinuousMultivariateDistribution, args)
    log_lambdas = zeros(Float64, atoms.num_atoms+1)
    logp_theta_new = logpdf(p, theta_new)

    mu = vec(mean(atoms.theta, 2))
    sigma = vec(std(atoms.theta, 2, corrected=false, mean=mu))
    q = construct_mixture_distribution(mu, sigma, args)
    log_lambdas[atoms.num_atoms+1] = logpdf(q, theta_new) - logp_theta_new

    for j = 1:atoms.num_atoms
        atoms.theta[:, j], theta_new = theta_new, atoms.theta[:, j]
        mu = vec(mean(atoms.theta, 2))
        sigma = vec(std(atoms.theta, 2, corrected=false, mean=mu))
        q = construct_mixture_distribution(mu, sigma, args)
        logq_theta_j = logpdf(q, theta_new)
        log_lambdas[j] = logq_theta_j - atoms.logp_theta[j]
        atoms.theta[:, j], theta_new = theta_new, atoms.theta[:, j]
    end

    # Subtract the max in log probability domain, and exponentiate back to probabilities
    log_lambdas -= maximum(log_lambdas)
    return exp.(log_lambdas), logp_theta_new
end


function construct_mixture_distribution(mu, sigma, args)
    if length(args["mixture_coef"]) == 1
        q = MvNormal(mu, sigma)
        return q
    end

    mixture_params = []
    for i in 1:length(args["mixture_coef"])
        push!(mixture_params, (mu, sqrt(args["mixture_coef"][i])*sigma))
    end
    q = MixtureModel(MvNormal, mixture_params, args["mixture_prob"])
    return q
end


end
