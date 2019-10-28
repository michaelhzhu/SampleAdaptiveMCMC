module MHM


using Distributions
using AtomsM, Models
export MH, MH_step, MH_step_propose


function MH(atoms, K, p, q0, q, args)

    if atoms.num_atoms != 1; error(); end

    # Initialize the state
    N = atoms.num_atoms
    theta_set = rand(q0, N)
    for i in 1:N
        save_atom(atoms, theta_set[:,i], logpdf(p, theta_set[:,i]), i)
    end
    
    for k = 1:K
        MH_step(atoms, p, q, 1)
        snapshot(atoms)
        if k % args["print_every"] == 0
            print(k, ": ", logpdf(p, vec(atoms.theta)), "\t")
            print_errors(p, atoms)
            println()
            flush(STDOUT)
        end
    end
end


function MH_step(atoms, p, q, n)
    # Run 1 step of MH for atom n
    theta_new = MH_step_propose(atoms.theta[:,n], q)
    logp_theta_new = logpdf(p, theta_new)
    log_acc_ratio = logp_theta_new - atoms.logp_theta[n]
    if log_acc_ratio >= 0 || rand(Float64) < exp(log_acc_ratio)
        save_atom(atoms, theta_new, logp_theta_new, n)
        atoms.last_acc = true
    else
        atoms.last_acc = false
    end
end


function MH_step_propose(x::Array{Float64,1}, q::ContinuousMultivariateDistribution)
    return rand(q) + x
end


end
