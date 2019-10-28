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

    mixture_coef = args["mixture_coef"]
    mixture_prob = args["mixture_prob"]
    index = rand(Categorical(mixture_prob))

    q = MvNormal(mu, sqrt(mixture_coef[index])*sigma)
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

    theta = atoms.theta
    dim_theta = size(theta, 1)
    num_atoms = size(theta, 2)

    m = mean(theta, 2)
    S_sqr = sum((theta .- m).^2, 2)
    v = S_sqr/atoms.num_atoms

    # Calculate all means and variances for the substitution procedure
    means = atoms.means
    variances = atoms.variances
    @. means = (theta_new .- theta)/atoms.num_atoms
    @. variances = (S_sqr .+ (theta_new .- m).^2 .- (theta .- m).^2)/atoms.num_atoms - means.^2

    # Calculate log Gaussian densities
    @. means = (theta - m - means).^2./variances
    @. variances = log.(variances)
    log_q_term1 = -sum(means, 1)/2
    log_q_term2 = -sum(variances, 1)/2
    log_q_new_term1 = -sum((theta_new - m).^2./v)/2
    log_q_new_term2 = -sum(log.(v))/2

    # Calculate log Gaussian scale-mixture densities
    mixture_coef = args["mixture_coef"]
    mixture_prob = args["mixture_prob"]
    log_q_mixture = atoms.log_q_mixture
    log_q_new_mixture = atoms.log_q_new_mixture

    for i in 1:length(mixture_coef)
        mix_c = mixture_coef[i]
        mix_p = mixture_prob[i]

        log_q_new_mixture[i] = log_q_new_term1/mix_c + log_q_new_term2 - dim_theta*log(mix_c)/2 + log(mix_p)
        log_q_mixture[:, i] = log_q_term1/mix_c + log_q_term2 - dim_theta*log(mix_c)/2 + log(mix_p)
    end

    log_q_new = logsumexp(log_q_new_mixture)
    log_q = logsumexp(log_q_mixture, 2)
    
    # Calculate the substitution probabilities lambdas
    log_lambdas = zeros(Float64, atoms.num_atoms+1)
    logp_theta_new = logpdf(p, theta_new)
    log_lambdas[1:atoms.num_atoms] = vec(log_q) - atoms.logp_theta
    log_lambdas[atoms.num_atoms+1] = log_q_new - logp_theta_new
    
    # Subtract the max in log probability domain, and exponentiate back to probabilities
    log_lambdas = log_lambdas - maximum(log_lambdas)
    return exp.(log_lambdas), logp_theta_new
end


function logsumexp(u, axes)
   m = maximum(u, axes)
   return m .+ log.(sum(exp.(u .- m), axes))
end

function logsumexp(u)
   m = maximum(u)
   return m .+ log.(sum(exp.(u .- m)))
end


end
