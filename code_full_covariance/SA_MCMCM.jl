module SA_MCMCM


using Distributions, PDMats
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
    cov_matrix = cov(atoms.theta, 2)

    q = MvNormal(mu, cov_matrix)
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

    log_q_all = zeros(num_atoms+1)
    get_all_log_q_probs(theta, theta_new, num_atoms, log_q_all)

    log_q = log_q_all[1:num_atoms]
    log_q_new = log_q_all[num_atoms+1]

    
    # Calculate the substitution probabilities lambdas
    log_lambdas = zeros(Float64, atoms.num_atoms+1)
    logp_theta_new = logpdf(p, theta_new)
    log_lambdas[1:atoms.num_atoms] = vec(log_q) - atoms.logp_theta
    log_lambdas[atoms.num_atoms+1] = log_q_new - logp_theta_new
    
    # Subtract the max in log probability domain, and exponentiate back to probabilities
    log_lambdas = log_lambdas - maximum(log_lambdas)
    return exp.(log_lambdas), logp_theta_new
end


# sumlogdiag_scaled(Σ, d, div_factor) = sum(log.(diag(Σ)/div_factor))
function sumlogdiag_scaled(Σ, d, div_factor)
    c = 0.0
    for i=1:d
        c += log(Σ[i,i])
    end
    return c - d*div_factor
end

function logpdfnormal_scaled(x, S, div_factor, d)
    BLAS.trsv!('U', 'T', 'N', S, x)
    -((BLAS.nrm2(d, x, 1)*div_factor)^2 + 2sumlogdiag_scaled(S,d,div_factor) + d*log(2pi))/2
end

function get_all_log_q_probs(theta, theta_new, N, u)
    mu = vec(mean(theta, 2))
    C1 = (theta .- mu)*(theta .- mu).'
    C1 += (theta_new - mu)*(theta_new - mu).'
    LinAlg.LAPACK.potrf!('U', C1)

    D = length(theta_new)
    chol_new = copy(C1)
    chol_wrapper = LinAlg.Cholesky(chol_new, 'U')
    LinAlg.lowrankdowndate!(chol_wrapper, (theta_new - mu))
    u[N+1] = logpdfnormal_scaled(theta_new - mu, chol_new, sqrt(N-1), D)

    temp1 = similar(mu)
    temp2 = similar(mu)
    temp3 = similar(mu)
    temp4 = similar(mu)
    @views for j in 1:N
        copy!(chol_new, C1)
        copy!(temp1, theta[:,j])
        LinAlg.axpy!(-1, mu, temp1)
        copy!(temp3, temp1)
        LinAlg.lowrankdowndate!(chol_wrapper, temp1)

        copy!(temp2, theta_new)
        LinAlg.axpy!(-1, theta[:,j], temp2)
        LinAlg.BLAS.scal!(D, 1.0/sqrt(N), temp2, 1)
        copy!(temp4, temp2)
        LinAlg.lowrankdowndate!(chol_wrapper, temp2)

        LinAlg.axpy!(-1.0/sqrt(N), temp4, temp3)
        u[j] = logpdfnormal_scaled(temp3, chol_new, sqrt(N-1), D)
    end
end


end
