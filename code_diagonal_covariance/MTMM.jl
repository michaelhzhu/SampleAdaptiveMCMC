module MTMM


using Distributions, StatsFuns
using AtomsM, Models
using MHM
export MTM


function MTM(atoms, K, p, q0, q, args)

    if atoms.num_atoms != 1; error(); end

    # Initialize the state
    N = atoms.num_atoms
    theta_set = rand(q0, N)
    for i in 1:N
        save_atom(atoms, theta_set[:,i], logpdf(p, theta_set[:,i]), i)
    end

    for k = 1:K
        MTM_step(atoms, p, q, 1, args["mtm_num_proposals"])
        snapshot(atoms)
        if k % args["print_every"] == 0
            print(k, ": ", logpdf(p, vec(atoms.theta)), "\t")
            print_errors(p, atoms)
            println()
            flush(STDOUT)
        end
    end
end


function MTM_step(atoms, p, q, n, num_proposals)

    dim_theta = size(atoms.theta, 1)
    Y = zeros(dim_theta, num_proposals)
    X = zeros(dim_theta, num_proposals)
    logp_Y = zeros(num_proposals)
    logp_X = zeros(num_proposals)

    for k in 1:num_proposals
        Y[:, k] = MH_step_propose(atoms.theta[:,n], q)
        logp_Y[k] = logpdf(p, Y[:, k])
    end

    logp_Y_offset = maximum(logp_Y)
    w_Y = exp.(logp_Y - logp_Y_offset)
    dist = Categorical(w_Y/sum(w_Y))
    index = rand(dist)

    for k in 1:num_proposals-1
        X[:, k] = MH_step_propose(Y[:, index], q)
        logp_X[k] = logpdf(p, X[:, k])
    end
    X[:, num_proposals] = atoms.theta[:, n]
    logp_X[num_proposals] = atoms.logp_theta[n]

    logw_offset = max(logp_Y_offset, maximum(logp_X))

    acc_ratio = sum(exp.(logp_Y - logw_offset))/sum(exp.(logp_X - logw_offset))
    if rand(Float64) < acc_ratio
        save_atom(atoms, Y[:, index], logp_Y[index], n)
        atoms.last_acc = true
    else
        atoms.last_acc = false
    end
end


end
