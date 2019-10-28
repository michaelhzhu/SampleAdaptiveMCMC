module Model_LogisticReg


using Distributions, StatsFuns
using AtomsM
export get_model, print_errors


type LogisticReg <: ContinuousMultivariateDistribution

    D::Int
    x_train::Array{Float64,2}
    x_test::Array{Float64,2}
    y_train::Array{Float64,1}
    y_test::Array{Float64,1}
    sigma::Float64

    LogisticReg(x_train, x_test, y_train, y_test, sigma) = new(size(x_train,2), x_train, x_test, y_train, y_test, sigma)
end


function Distributions.length(data::LogisticReg)
    return data.D
end


# https://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w8a_bayes_logistic_regression_laplace.pdf
function Distributions._logpdf{T<:Real}(data::LogisticReg, new_param::AbstractVector{T})
    theta = new_param

    z = data.y_train .* (data.x_train * theta)
    logp = -sum(log1pexp.(-z))
    logp += logpdf(MvNormal(data.D, data.sigma), theta)
    return logp
end


function calc_error(data, theta, x_train, y_train)
    score = x_train * theta
    predictions = score .> 0.0
    ground_truth = y_train .> 0.0
    accuracy = mean(predictions .== ground_truth)
    return 1.0 - accuracy
end

function print_errors(data::LogisticReg, atoms::Atoms)
    theta = mean(atoms.theta, 2)

    z_train = data.y_train .* (data.x_train * theta)
    logp_train = -mean(log1pexp.(-z_train))
    z_test = data.y_test .* (data.x_test * theta)
    logp_test = -mean(log1pexp.(-z_test))
    err_train = calc_error(data, theta, data.x_train, data.y_train)
    err_test = calc_error(data, theta, data.x_test, data.y_test)

    current_acc_rate = atoms.current_acc/atoms.args["print_every"]
    atoms.current_acc = 0

    print("$(logp_train)\t$(logp_test)\t$(err_train)\t$(err_test)\t$(current_acc_rate)")
end


function get_model(sigma)
    x_train = readdlm("data_logistic_reg/x_train.txt")
    x_test = readdlm("data_logistic_reg/x_test.txt")
    y_train = vec(readdlm("data_logistic_reg/y_train.txt"))
    y_test = vec(readdlm("data_logistic_reg/y_test.txt"))

    p = LogisticReg(x_train, x_test, y_train, y_test, sigma)
    return p, p.D
end


end
