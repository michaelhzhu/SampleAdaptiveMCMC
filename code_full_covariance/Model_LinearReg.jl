module Model_LinearReg


using Distributions, StatsFuns
using AtomsM
export generate_data, get_model, print_errors


type LinearReg <: ContinuousMultivariateDistribution

    D::Int
    x_train::Array{Float64,2}
    x_test::Array{Float64,2}
    y_train::Array{Float64,1}
    y_test::Array{Float64,1}
    theta_true::Array{Float64, 1}
    sigma::Float64
    lambda::Float64

    LinearReg(x_train, x_test, y_train, y_test, theta_true, sigma, lambda) = new(size(x_train,2), x_train, x_test, y_train, y_test, theta_true, sigma, lambda)
end


function Distributions.length(data::LinearReg)
    return data.D
end


function Distributions._logpdf{T<:Real}(data::LinearReg, new_param::AbstractVector{T})
    w = new_param
    
    logp = logpdf(MvNormal(data.x_train * w, data.sigma), data.y_train)
    logp += sum(logpdf.(Laplace(0, data.lambda), w))
    return logp
end


function print_errors(data::LinearReg, atoms::Atoms)
    theta = mean(atoms.theta, 2)
    
    y_train_pred = data.x_train * theta
    y_test_pred = data.x_test * theta

    mse_train = mean((y_train_pred .- data.y_train).^2)
    mse_test = mean((y_test_pred .- data.y_test).^2)
    
    mss_train = mean((mean(data.y_train) .- data.y_train).^2)
    mss_test = mean((mean(data.y_test) .- data.y_test).^2)
    r2_train = 1.0 - mse_train/mss_train
    r2_test = 1.0 - mse_test/mss_test
    
    rmse_w = sqrt(mean((theta .- data.theta_true).^2))
    abse_w = mean(abs.(theta .- data.theta_true))

    current_acc_rate = atoms.current_acc/atoms.args["print_every"]
    atoms.current_acc = 0

    print("$(r2_train)\t$(r2_test)\t$(rmse_w)\t$(abse_w)\t$(mse_train)\t$(mse_test)\t$(current_acc_rate)")
end


function generate_data(D, N, sigma, lambda, folder)
    file_x_train = string(folder, "/x_train_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt")
    file_x_test = string(folder, "/x_test_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt")
    file_y_train = string(folder, "/y_train_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt")
    file_y_test = string(folder, "/y_test_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt")
    file_params = string(folder, "/true_parameters_", D, "_", N, "_", lambda, "_", sigma, "_", ".txt")

    if isfile(file_x_train); return; end

    theta_true = rand(Laplace(0, lambda), D)
    
    x = rand(Normal(), N, D-1)
    for j in 1:D-1
        x[:,j] *= (j+1)/2
    end
    x = hcat(ones(size(x)[1]), x)
    
    y = rand(MvNormal(x * theta_true, sigma))

    split_n = Int(0.8*N)
    x_train = x[1:split_n, :]
    y_train = y[1:split_n]
    x_test = x[split_n+1:end, :]
    y_test = y[split_n+1:end]

    writedlm(file_x_train, x_train)
    writedlm(file_y_train, y_train)
    writedlm(file_x_test, x_test)
    writedlm(file_y_test, y_test)
    writedlm(file_params, theta_true)
end


function get_model(D, N, sigma, lambda, folder)
    x_train = readdlm(string(folder, "/x_train_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt"))
    x_test = readdlm(string(folder, "/x_test_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt"))
    y_train = vec(readdlm(string(folder, "/y_train_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt")))
    y_test = vec(readdlm(string(folder, "/y_test_", D, "_", N, "_", sigma, "_", lambda, "_", ".txt")))
    theta_true = vec(readdlm(string(folder, "/true_parameters_", D, "_", N, "_", lambda, "_", sigma, "_", ".txt")))

    p = LinearReg(x_train, x_test, y_train, y_test, theta_true, sigma, lambda)
    return p, p.D
end


end
