module AtomsM


export Atoms
export save_atom, snapshot


type Atoms
    # theta: [dim_theta, num_atoms]
    # logp_theta: [num_atoms] cache p(theta)
    # theta_averaged_history: [dim_theta, K-kappa] history of theta, averaged over the atoms in the state, after burn-in
    #                         only saved if args["save_theta_averaged_history"] is true

    # args
    # num_atoms: number of atoms in the state
    # K: total number of MCMC iterations
    # kappa: number of burn-in iterations
    # k: current MCMC iteration to support saving theta_averaged_history

    # total_acc: total number of accepted moves for each atom after burn-in
    # current_acc: number of accepted moves for each atom in a block of args["print_every"] iterations
    # last_acc: indicator of whether or not each atom accepted a move the last iteration

    # means, variances, log_q_mixture, log_q_new_mixture
    # are pre-allocated so the arrays can be reused in each iteration of SA-MCMC

    theta::Array{Float64,2}
    logp_theta::Array{Float64,1}
    theta_averaged_history

    args
    num_atoms::Int
    K::Int
    kappa::Int
    k::Int

    total_acc::Int
    current_acc::Int
    last_acc::Bool

    means::Array{Float64, 2}
    variances::Array{Float64, 2}
    log_q_mixture::Array{Float64, 2}
    log_q_new_mixture::Array{Float64, 1}

    Atoms(num_atoms::Int, dim_theta::Int, K::Int, kappa::Int, args) = begin
        theta = zeros(Float64, dim_theta, num_atoms)
        logp_theta = zeros(Float64, num_atoms)
        if args["save_theta_averaged_history"]
            theta_averaged_history = zeros(Float64, dim_theta, K-kappa)
            println("Theta history dimensions = $(size(theta_averaged_history))\n")
        else
            theta_averaged_history = nothing
        end

        new(theta, logp_theta, theta_averaged_history,
            args, num_atoms, K, kappa, 1,
            0, 0, 0,
            zeros(dim_theta, num_atoms), zeros(dim_theta, num_atoms),
            zeros(num_atoms, length(args["mixture_coef"])), zeros(length(args["mixture_coef"])))
    end
end


function Base.show(io::IO, atoms::Atoms)
    println(io, transpose(mean(atoms.theta, 2)))
end


# Saves (theta, logp_theta) into the atom set at index i
function save_atom(atoms, theta, logp_theta, i)
    atoms.theta[:, i] = theta
    atoms.logp_theta[i] = logp_theta
end


# Updates theta_averaged_history and acceptance statistics
function snapshot(atoms)
    atoms.current_acc += atoms.last_acc

    if atoms.k > atoms.kappa
        if atoms.theta_averaged_history !== nothing
            atoms.theta_averaged_history[:, atoms.k-atoms.kappa] = mean(atoms.theta, 2)
        end
        atoms.total_acc += atoms.last_acc
    end
    atoms.k += 1
end


end
