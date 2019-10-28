workspace()
push!(LOAD_PATH, ".")

BLAS.set_num_threads(1)

module MainM


using ArgParse, Distributions, MAT
using AtomsM, Model_LinearReg, Model_LogisticReg
using SA_MCMCM, MHM, AdaptiveMHM, MTMM


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin

        # K: total number of MCMC iterations
        # kappa: number of burn-in iterations
        # print_every: how often to print statistics
        # save_theta_averaged_history: whether to save history of theta, averaged over the atoms in the state, after burn-in
        # initialize_position_from_file: if true, initializes position from data_logistic_reg/initial_position.txt
        "--K"
            default = 1100000; arg_type = Int
        "--kappa"
            default = 100000; arg_type = Int
        "--print_every"
            default = 10000; arg_type = Int
        "--save_theta_averaged_history"
            action = :store_true
        "--initialize_position_from_file"
            action = :store_true

        # run_sa: Sample Adaptive MCMC
        # N: number of atoms in the state
        # mixture_coef: mixture coefficients in the Gaussian scale-mixture family of proposal distributions
        # mixture_prob: mixture probabilities in the Gaussian scale-mixture family of proposal distributions
        # q0_std: standard deviation of initialization distribution
        "--run_sa"
            action = :store_true
        "--N"
            default = 40; arg_type = Int
        "--mixture_coef"
            default = "1.0,0.5,2.0"
        "--mixture_prob"
            default = "0.3334,0.3333,0.3333"
        "--q0_std"
            default = 1.0; arg_type = Float64

        # run_mh: Metropolis-Hastings
        # q_std: standard deviation of proposal distribution
        "--run_mh"
            action = :store_true
        "--q_std"
            default = 0.016; arg_type = Float64

        # run_am: Adaptive Metropolis-Hastings
        # adaptive_scale: scale parameter of empirical covariance matrix
        "--run_am"
            action = :store_true
        "--adaptive_scale"
            default = 0.8; arg_type = Float64

        # run_mtm: Multiple-Try Metropolis
        # mtm_num_proposals: number of proposals
        "--run_mtm"
            action = :store_true
        "--mtm_num_proposals"
            default = 3; arg_type = Int

        # model: "logistic_reg", "linear_reg"
        # trial: trial number (used for setting random seed)
        "--model"
            default = "logistic_reg"; arg_type = String
        "--trial"
            default = 1; arg_type = Int

        # logistic_reg model
        # logistic_reg_sigma: standard deviation of Gaussian prior
        # logistic_reg_name: name for setting random seed
        "--logistic_reg_sigma"
            default = 1.0; arg_type = Float64
        "--logistic_reg_name"
            default = "census"; arg_type = String
        
        # linear_reg model
        # linear_reg_dataset_size: number of data points
        # linear_reg_dim: dimension (number of regression coefficients)
        # linear_reg_sigma: standard deviation of Gaussian distribution
        # linear_reg_lambda: Laplace prior
        "--linear_reg_dataset_size"
            default = 10000; arg_type = Int
        "--linear_reg_dim"
            default = 10; arg_type = Int
        "--linear_reg_sigma"
            default = 10.0; arg_type = Float64
        "--linear_reg_lambda"
            default = 1.0; arg_type = Float64
    end

    a = parse_args(s)
    a["mixture_coef_str"] = a["mixture_coef"]
    a["mixture_prob_str"] = a["mixture_prob"]
    a["mixture_coef"] = parse_list_arg(a["mixture_coef"])
    a["mixture_prob"] = parse_list_arg(a["mixture_prob"])
    return a, a["N"], a["q0_std"], a["q_std"], a["K"], a["kappa"]
end


function parse_list_arg(s)
    str_list = split(s, ",")
    float_list = map(x->parse(Float64,x), str_list)
    return float_list
end


function main()

    const args, N, q0_std, q_std, K, kappa = parse_commandline()
    
    if args["model"] == "logistic_reg"
        model_str = args["logistic_reg_name"]
        p, dim_theta = Model_LogisticReg.get_model(args["logistic_reg_sigma"])
    elseif args["model"] == "linear_reg"
        model_str = "$(args["model"])_$(args["linear_reg_dataset_size"])_$(args["linear_reg_dim"])_$(args["linear_reg_sigma"])_$(args["linear_reg_lambda"])"
        println(model_str)
        srand(hash(model_str))
        if !isdir("data_linear_reg"); mkpath("data_linear_reg"); end
        Model_LinearReg.generate_data(args["linear_reg_dim"], args["linear_reg_dataset_size"], args["linear_reg_sigma"], args["linear_reg_lambda"], "data_linear_reg")
        p, dim_theta = Model_LinearReg.get_model(args["linear_reg_dim"], args["linear_reg_dataset_size"], args["linear_reg_sigma"], args["linear_reg_lambda"], "data_linear_reg")
    else
        error("Model not recognized")
    end

    if args["run_sa"]
        args["s"] = "$(model_str)_sa_$(N)_$(q0_std)_$(args["mixture_coef_str"])_$(args["mixture_prob_str"])_$(args["trial"])"
    elseif args["run_mh"]
        args["s"] = "$(model_str)_mh_$(q_std)_$(args["trial"])"
    elseif args["run_am"]
        args["s"] = "$(model_str)_am_$(args["adaptive_scale"])_$(q_std)_$(args["trial"])"
    elseif args["run_mtm"]
        args["s"] = "$(model_str)_mtm_$(args["mtm_num_proposals"])_$(q_std)_$(args["trial"])"
    else
        return
    end
    println(args["s"])
    srand(hash(args["s"]))


    if args["initialize_position_from_file"]
        initial_pos = vec(readdlm("data_logistic_reg/initial_position.txt"))
    else
        initial_pos = zeros(dim_theta)
    end
    
    q0_SA = MvNormal(initial_pos, q0_std)
    q0_MH = MvNormal(initial_pos, q_std)
    q = MvNormal(zeros(dim_theta), q_std)


    if args["run_sa"]
        atoms = Atoms(N, dim_theta, K, kappa, args)
        @time SA_MCMC(atoms, K, p, q0_SA, args)
    end

    if args["run_mh"]
        atoms = Atoms(1, dim_theta, K, kappa, args)
        @time MH(atoms, K, p, q0_MH, q, args)
    end

    if args["run_am"]
        atoms = Atoms(1, dim_theta, K, kappa, args)
        @time AdaptiveMH(atoms, kappa, K, p, q0_MH, q, args)
    end

    if args["run_mtm"]
        atoms = Atoms(1, dim_theta, K, kappa, args)
        @time MTM(atoms, K, p, q0_MH, q, args)
    end


    num_acc = atoms.total_acc
    num_iters = atoms.k-atoms.kappa-1
    println()
    println("Number accepted = ", num_acc)
    println("Number iterations = ", num_iters)
    println("Acceptance rate = ", num_acc/num_iters)

    if args["save_theta_averaged_history"]
        save_dir = "saved"
        if !isdir(save_dir); mkpath(save_dir); end
        mat_filename = string(save_dir, "/", args["s"], ".mat")
        matwrite(mat_filename, Dict("theta_averaged_history" => atoms.theta_averaged_history))
    end

end


main()

end
