"""
    Experiments on real datasets:
        Regularized Gauss-Newton for learning overparameterized neural networks.
"""

include(joinpath(@__DIR__, "..", "src/NNOptimization.jl"))
include(joinpath(@__DIR__, "utils/plot-utils.jl"))
include(joinpath(@__DIR__, "utils/problem-utils.jl"))

using SelfConcordantSmoothOptimization
using JLD

filedir = pwd() * "/experiments/"

function train_accuracy(model, x)
    Φ_model(X) = model.out_fn(X, x)'
    accuracy(Φ_model, model.A, model.y')
end
function test_accuracy(model, x)
    Φ_model(X) = model.out_fn(X, x)'
    accuracy(Φ_model, model.Atest, model.ytest')
end
function activation_stability(model, result, n1)
    m, n0 = size(model.A)
    n1n0 = n1*n0
    θ0 = model.x0
    θ = result.x
    u0 = reshape(θ0[1:n1n0], (n1, n0))
    u = reshape(θ[1:n1n0], (n1, n0))
    pre_act0 = u0*model.A'
    pre_act = u*model.A'
    sl = sum(sign.(pre_act) .== sign.(pre_act0))/(m*n1)
    return sl
end

function get_t_model_dir(problem_name)
    mnist_t_model_dir = filedir*"tmp/mnist_teacher_model.bson"
    fashionmnist_t_model_dir = filedir*"tmp/fashionmnist_teacher_model.bson"
    avila_t_model_dir = filedir*"tmp/avila_teacher_model.bson"
    pendigits_t_model_dir = filedir*"tmp/pendigits_teacher_model.bson"
    letter_t_model_dir = filedir*"tmp/letter_teacher_model.bson"

    if problem_name in ["mnist", "mnist-t-s"]
        t_model_dir = mnist_t_model_dir
    elseif problem_name in ["fashionmnist", "fashionmnist-t-s"]
        t_model_dir = fashionmnist_t_model_dir
    elseif problem_name in ["pendigits", "pendigits-t-s"]
        t_model_dir = pendigits_t_model_dir
    elseif problem_name in ["letter", "letter-t-s"]
        t_model_dir = letter_t_model_dir
    elseif problem_name in ["avila", "avila-t-s"]
        t_model_dir = avila_t_model_dir
    end

    t_model_dir
end

function activation_stability(problem_name, params0, params, n1, sl)
    t_model_dir = get_t_model_dir(problem_name)
    (_, _, model, _, _, _) = gen_problem_data(problem_name=problem_name, use_prox=false, t_model_dir=t_model_dir)
    m, n0 = size(model.A)
    n1n0 = n1*n0
    u0vec = params0[1:n1n0]
    uvec = params[1:n1n0]
    nz_u0 = n1n0 - cardcal(u0vec, 0.999)
    nz_u = n1n0 - cardcal(uvec, 0.999)
    if nz_u ≥ nz_u0
        nz_u1 = nz_u - nz_u0
    else
        nz_u1 = 0
    end
    nz_u1 = 100*(nz_u1/(m*n1))
    return nz_u, nz_u1, sl + nz_u1
end

function save_act_stability(resultsLbda, resultsLbdaTS, resultsMu, resultsMuTS)
    n_lambdas = length(resultsLbda["params"])
    n_mus = length(resultsMu["params"])

    params0Lbda = resultsLbda["params0"]
    paramsLbda = resultsLbda["params"]
    slsLbda = resultsLbda["sls"]

    params0LbdaTS = resultsLbdaTS["params0"]
    paramsLbdaTS = resultsLbdaTS["params"]
    slsLbdaTS = resultsLbdaTS["sls"]

    params0Mu = resultsMu["params0"]
    paramsMu = resultsMu["params"]
    slsMu = resultsMu["sls"]

    params0MuTS = resultsMuTS["params0"]
    paramsMuTS = resultsMuTS["params"]
    slsMuTS = resultsMuTS["sls"]

    nzuallLbda = Vector{Float64}(undef, n_lambdas)
    nzuLbda = Vector{Float64}(undef, n_lambdas)
    slspLbda = Vector{Float64}(undef, n_lambdas)
    nzuallLbdaTS = Vector{Float64}(undef, n_lambdas)
    nzuLbdaTS = Vector{Float64}(undef, n_lambdas)
    slspLbdaTS = Vector{Float64}(undef, n_lambdas)
    nzuallMu = Vector{Float64}(undef, n_mus)
    nzuMu = Vector{Float64}(undef, n_mus)
    slspMu = Vector{Float64}(undef, n_mus)
    nzuallMuTS = Vector{Float64}(undef, n_mus)
    nzuMuTS = Vector{Float64}(undef, n_mus)
    slspMuTS = Vector{Float64}(undef, n_mus)

    algo="ggn"
    problem_name="mnist"
    activation ="relu"
    n1=512
    batch_size=16
    for i in 1:n_lambdas
        nzuallLbda[i], nzuLbda[i], slspLbda[i] = activation_stability(problem_name, params0Lbda[i], paramsLbda[i], n1, slsLbda[i])
    end
    for i in 1:n_mus
        nzuallMu[i], nzuMu[i], slspMu[i] = activation_stability(problem_name, params0Mu[i], paramsMu[i], n1, slsMu[i])
    end
    JLD.save(filedir*"tmp/slsp_$(problem_name)_"*activation*"_$(algo)_$(n1)_$(batch_size).jld", "nzuallLbda", nzuallLbda, "nzuLbda", nzuLbda, "slspLbda", slspLbda, "nzuallMu", nzuallMu, "nzuMu", nzuMu, "slspMu", slspMu)

    problem_name="mnist-t-s"
    activation ="silu"
    n1=1024
    for i in 1:n_lambdas
        nzuallLbdaTS[i], nzuLbdaTS[i], slspLbdaTS[i] = activation_stability(problem_name, params0LbdaTS[i], paramsLbdaTS[i], n1, slsLbdaTS[i])
    end
    for i in 1:n_mus
        nzuallMuTS[i], nzuMuTS[i], slspMuTS[i] = activation_stability(problem_name, params0MuTS[i], paramsMuTS[i], n1, slsMuTS[i])
    end
    JLD.save(filedir*"tmp/slsp_$(problem_name)_"*activation*"_$(algo)_$(n1)_$(batch_size).jld", "nzuallLbdaTS", nzuallLbdaTS, "nzuLbdaTS", nzuLbdaTS, "slspLbdaTS", slspLbdaTS, "nzuallMuTS", nzuallMuTS, "nzuMuTS", nzuMuTS, "slspMuTS", slspMuTS)
    
end


function RunMNISTExpLambda(;algo="ggn",problem_name="mnist",activation ="relu", n1=512, n1_star=16, batch_size=16, max_epoch=2, save_results=true)

    lambdas = 10.0.^(-6:1:0)
    n_lambdas = length(lambdas)
    κn = 1/sqrt(n1)
    μ = 1/κn

    metrics = Dict()
    metrics["train_accuracy"] = train_accuracy
    metrics["test_accuracy"] = test_accuracy

    train_accuracies = Vector{Float64}(undef, n_lambdas)
    test_accuracies = Vector{Float64}(undef, n_lambdas)
    train_losses = Vector{Float64}(undef, n_lambdas)
    train_losses_reg = Vector{Float64}(undef, n_lambdas)
    test_losses = Vector{Float64}(undef, n_lambdas)
    nnzs = Vector{Float64}(undef, n_lambdas)
    sls = Vector{Float64}(undef, n_lambdas)
    slps = Vector{Tuple{Int64, Float64, Float64}}(undef, n_lambdas)
    params0 = Vector{Vector{Float64}}(undef, n_lambdas)
    params = Vector{Vector{Float64}}(undef, n_lambdas)
    t_model_dir = get_t_model_dir(problem_name)
    t = 0
    for i in axes(lambdas,1)
        λ = lambdas[i]
        print("="^20*"\n")
        @show λ
        (_, testdata, model, optimizer, reg_name, hμ) = gen_problem_data(problem_name=problem_name, n1=n1, n1_star=n1_star, λ=λ, μ=μ, activation=activation, κn=κn, κW=1.0, algo=algo, use_prox=false, t_model_dir=t_model_dir, max_epochs_t_s=250)
        result = iterate!(optimizer, model, reg_name, hμ; batch_size=batch_size, α=1, max_epoch=max_epoch, f_tol=1e-8, metrics=metrics, verbose=3)
        sl = 100*activation_stability(model, result, n1)
        slp = activation_stability(problem_name, model.x0, result.x, n1, sl)
        @show sl slp

        t += result.times[end]

        train_accuracies[i] = result.metricvals["train_accuracy"][end]
        test_accuracies[i] = result.metricvals["test_accuracy"][end]
        train_losses[i] = result.fval[end]
        train_losses_reg[i] = result.obj[end]
        test_losses[i] = result.fvaltest[end]
        nnzs[i] = cardcal(result.x, 0.999)
        sls[i] = sl
        slps[i] = slp
        params[i] = result.x
        params0[i] = model.x0
    end
    
    if save_results
        JLD.save(filedir*"tmp/lbda_$(problem_name)_"*activation*"_$(algo)_$(n1)_$(batch_size).jld", "times", t, "lambdas", lambdas, "train_accuracies", train_accuracies, "test_accuracies", test_accuracies, "train_losses", train_losses, "train_losses_reg", train_losses_reg, "test_losses", test_losses, "nnzs", nnzs, "sls", sls, "slps", slps, "params", params, "params0", params0)
    end

end

function RunMNISTExpMu(;algo="ggn", problem_name="mnist", activation ="relu", n1=512, n1_star=16, batch_size=16, max_epoch=2, save_results=true)

    if problem_name in ["mnist", "mnist-t-s"]
        mus = 10.0.^(-3:1:3)
    else
        mus = 10.0.^(-6:1:0)
    end
    n_mus = length(mus)
    κn = 1/sqrt(n1)
    λ = 1e-4

    metrics = Dict()
    metrics["train_accuracy"] = train_accuracy
    metrics["test_accuracy"] = test_accuracy

    train_accuracies = Vector{Float64}(undef, n_mus)
    test_accuracies = Vector{Float64}(undef, n_mus)
    train_losses = Vector{Float64}(undef, n_mus)
    train_losses_reg = Vector{Float64}(undef, n_mus)
    test_losses = Vector{Float64}(undef, n_mus)
    nnzs = Vector{Float64}(undef, n_mus)
    sls = Vector{Float64}(undef, n_mus)
    slps = Vector{Tuple{Int64, Float64, Float64}}(undef, n_mus)
    params0 = Vector{Vector{Float64}}(undef, n_mus)
    params = Vector{Vector{Float64}}(undef, n_mus)
    t_model_dir = get_t_model_dir(problem_name)
    t = 0
    for i in axes(mus,1)
        μ = mus[i]
        print("="^20*"\n")
        @show μ
        (_, testdata, model, optimizer, reg_name, hμ) = gen_problem_data(problem_name=problem_name, n1=n1, n1_star=n1_star, λ=λ, μ=μ, activation=activation, κn=κn, κW=1.0, algo=algo, use_prox=false, t_model_dir=t_model_dir, max_epochs_t_s=250)
        result = iterate!(optimizer, model, reg_name, hμ; batch_size=batch_size, α=1, max_epoch=max_epoch, f_tol=1e-8, metrics=metrics, verbose=3)
        sl = 100*activation_stability(model, result, n1)
        slp = activation_stability(problem_name, model.x0, result.x, n1, sl)
        @show sl slp

        t += result.times[end]

        train_accuracies[i] = result.metricvals["train_accuracy"][end]
        test_accuracies[i] = result.metricvals["test_accuracy"][end]
        train_losses[i] = result.fval[end]
        train_losses_reg[i] = result.obj[end]
        test_losses[i] = result.fvaltest[end]
        nnzs[i] = cardcal(result.x, 0.999)
        sls[i] = sl
        slps[i] = slp
        params[i] = result.x
        params0[i] = model.x0
    end
    
    if save_results
        JLD.save(filedir*"tmp/mu_$(problem_name)_"*activation*"_$(algo)_$(n1)_$(batch_size).jld", "times", t, "mus", mus, "train_accuracies", train_accuracies, "test_accuracies", test_accuracies, "train_losses", train_losses, "train_losses_reg", train_losses_reg, "test_losses", test_losses, "nnzs", nnzs, "sls", sls, "slps", slps, "params", params, "params0", params0)
    end

end

function RunExpAny(;algo="ggn", problem_name="pendigits", activation ="relu", n1=128, batch_size=64, save_results=true)

    κn = 1/sqrt(n1)
    batch_size = batch_size
    if problem_name in ["avila", "letter"]
        μ = 10/κn
        if problem_name == "letter"
            batch_size = 8
        end
    elseif problem_name == "pendigits"
        μ = 0.001/κn
    else
        μ = 1/κn
    end

    if algo == "ggn"
        λ = 1e-4
        max_epoch = 50
    else
        λ = 0
        max_epoch = 2000
    end

    metrics = Dict()
    metrics["train_accuracy"] = train_accuracy
    metrics["test_accuracy"] = test_accuracy

    (_, _, model, optimizer, reg_name, hμ) = gen_problem_data(problem_name=problem_name, n1=n1, λ=λ, μ=μ, activation=activation, κn=κn, κW=1.0, algo=algo, use_prox=false)
    result = iterate!(optimizer, model, reg_name, hμ; batch_size=batch_size, α=1, max_epoch=max_epoch, f_tol=1e-8, metrics=metrics, verbose=3)
    sl = 100*activation_stability(model, result, n1)
    @show sl

    times = result.times
    train_accuracies = result.metricvals["train_accuracy"]
    test_accuracies = result.metricvals["test_accuracy"]
    train_losses = result.fval
    train_losses_reg = result.obj
    test_losses = result.fvaltest
    nnzs = cardcal(result.x, 0.999)
    params = result.x
    params0 = model.x0

    slp = activation_stability(problem_name, params0, params, n1, sl)
    @show slp
    
    if save_results
        JLD.save(filedir*"tmp/single_$(problem_name)_"*activation*"_$(algo)_$(n1)_$(batch_size).jld", "times", times,  "train_accuracies", train_accuracies, "test_accuracies", test_accuracies, "train_losses", train_losses, "train_losses_reg", train_losses_reg, "test_losses", test_losses, "nnzs", nnzs, "params", params, "params0", params0, "sl", sl, "slp", slp)
    end

end

function plotResultsLbdaMu(problem_name, resultsLbda, resultsLbdaTS, resultsMu, resultsMuTS, slsp, slspTS)
    xticks = [1e4,1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    yticks2 = [10,30,50,70,90,100]

    lambdas = resultsLbda["lambdas"]
    mus = resultsMu["mus"]

    test_lossesLbda = resultsLbda["test_losses"]
    test_lossesLbdaTS = resultsLbdaTS["test_losses"]
    test_lossesMu = resultsMu["test_losses"]
    test_lossesMuTS = resultsMuTS["test_losses"]

    test_accuraciesLbda = resultsLbda["test_accuracies"]
    test_accuraciesLbdaTS = resultsLbdaTS["test_accuracies"]
    test_accuraciesMu = resultsMu["test_accuracies"]
    test_accuraciesMuTS = resultsMuTS["test_accuracies"]

    train_accuraciesLbda = resultsLbda["train_accuracies"]
    train_accuraciesLbdaTS = resultsLbdaTS["train_accuracies"]
    train_accuraciesMu = resultsMu["train_accuracies"]
    train_accuraciesMuTS = resultsMuTS["train_accuracies"]

    nnzsLbda = resultsLbda["nnzs"]
    nnzsLbdaTS = resultsLbdaTS["nnzs"]
    nnzsMu = resultsMu["nnzs"]
    nnzsMuTS = resultsMuTS["nnzs"]

    slsLbda = resultsLbda["sls"]
    slsLbdaTS = resultsLbdaTS["sls"]
    slsMu = resultsMu["sls"]
    slsMuTS = resultsMuTS["sls"]

    slspLbda = slsp["slspLbda"]
    slspLbdaTS = slspTS["slspLbdaTS"]
    slspMu = slsp["slspMu"]
    slspMuTS = slspTS["slspMuTS"]

    paccLbda = plot(lambdas, train_accuraciesLbda, xscale=:log10, xticks=xticks, yticks=yticks2, label="training accuracy", ls=:auto, ylabel="%", xlabel="τ", leg=:bottom, dpi=600)
    plot!(lambdas, test_accuraciesLbda, label="test accuracy", ls=:auto)
    plot!(lambdas, slsLbda, label="T-I measure", ls=:solid)
    plot!(lambdas, slspLbda, label="T-I meas. (incl. # zeros)", ls=:dash)
    savefig(paccLbda, filedir * "figures/"*problem_name*"accuracyLbda.pdf")
    paccLbdaTS = plot(lambdas, train_accuraciesLbdaTS, xscale=:log10, xticks=xticks, yticks=yticks2, label="train accuracy", ls=:auto, ylabel="%", xlabel="τ", leg=:bottom, dpi=600)
    plot!(lambdas, test_accuraciesLbdaTS, label="test accuracy", ls=:auto)
    plot!(lambdas, slsLbdaTS, label="T-I measure", ls=:solid)
    plot!(lambdas, slspLbdaTS, label="T-I meas. (incl. # zeros)", ls=:dash)
    savefig(paccLbdaTS, filedir * "figures/"*problem_name*"accuracyLbdaTS.pdf")
    paccMu = plot(mus, train_accuraciesMu, xscale=:log10, xticks=xticks, yticks=yticks2, label="training accuracy", ls=:auto, ylabel="%", xlabel="μ", leg=:bottom, dpi=600)
    plot!(mus, test_accuraciesMu, label="test accuracy", ls=:auto)
    plot!(mus, slsMu, label="T-I measure", ls=:solid)
    plot!(mus, slspMu, xscale=:log10, xticks=xticks, label="T-I meas. (incl. # zeros)", ls=:dash)
    savefig(paccMu, filedir * "figures/"*problem_name*"accuracyMu.pdf")
    paccMuTS = plot(mus, train_accuraciesMuTS, xscale=:log10, xticks=xticks, yticks=yticks2, label="training accuracy", ls=:auto, ylabel="%", xlabel="μ", leg=:bottom, dpi=600)
    plot!(mus, test_accuraciesMuTS, label="test accuracy", ls=:auto)
    plot!(mus, slsMuTS, label="T-I measure", ls=:solid)
    plot!(mus, slspMuTS, label="T-I meas. (incl. # zeros)", ls=:dash)
    savefig(paccMuTS, filedir * "figures/"*problem_name*"accuracyMuTS.pdf")

    ################################################

    n0 = 28*28
    ny = 10

    n1 = 512
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    ax1 = axs[1].twinx()
    ax1.tick_params(axis="y", labelcolor="darkred")
    # ax1.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax1.set_yscale("linear")
    axs[1].tick_params(axis="y", labelcolor="black")
    axs[1].set_xlabel("μ")
    axs[1].set_ylabel("test loss", color="black", fontsize=20)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    ax2 = axs[2].twinx()
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax2.set_yscale("linear")
    axs[2].tick_params(axis="y", labelcolor="black")
    axs[2].set_xlabel("τ")
    # axs[2].set_ylabel("test loss", color="black", fontsize=20)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")

    axs[1].plot(mus, test_lossesMu, "black")
    ax1.plot(mus, (((n0*n1)+(n1*ny)) .- nnzsMu), "darkred")

    axs[2].plot(lambdas, test_lossesLbda, "black")
    ax2.plot(lambdas, (((n0*n1)+(n1*ny)) .- nnzsLbda), "darkred")
    
    fig.tight_layout()
    PyPlot.savefig(filedir * "figures/"*problem_name*"loss_combinedLbdaMu.pdf", dpi=600)

    n1 = 1024
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    ax1 = axs[1].twinx()
    ax1.tick_params(axis="y", labelcolor="darkred")
    # ax1.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax1.set_yscale("linear")
    axs[1].tick_params(axis="y", labelcolor="black")
    axs[1].set_xlabel("μ")
    axs[1].set_ylabel("test loss", color="black", fontsize=20)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")


    ax2 = axs[2].twinx()
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax2.set_yscale("linear")
    axs[2].tick_params(axis="y", labelcolor="black")
    axs[2].set_xlabel("τ")
    # axs[2].set_ylabel("test loss", color="black", fontsize=20)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")

    axs[1].plot(mus, test_lossesMuTS, "black")
    ax1.plot(mus, (((n0*n1)+(n1*ny)) .- nnzsMuTS), "darkred")

    axs[2].plot(lambdas, test_lossesLbdaTS, "black")
    ax2.plot(lambdas, (((n0*n1)+(n1*ny)) .- nnzsLbdaTS), "darkred")

    fig.tight_layout()
    PyPlot.savefig(filedir * "figures/"*problem_name*"loss_combinedLbdaMuTS.pdf", dpi=600)
end

function plotResultsLbdaMu(problem_name, resultsLbda, resultsMu, slsp)
    xticks = [1e4,1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    yticks2 = [20,40,60,80,100]

    lambdas = resultsLbda["lambdas"]
    mus = resultsMu["mus"]

    test_lossesLbda = resultsLbda["test_losses"]
    test_lossesMu = resultsMu["test_losses"]

    test_accuraciesLbda = resultsLbda["test_accuracies"]
    test_accuraciesMu = resultsMu["test_accuracies"]

    train_accuraciesLbda = resultsLbda["train_accuracies"]
    train_accuraciesMu = resultsMu["train_accuracies"]

    nnzsLbda = resultsLbda["nnzs"]
    nnzsMu = resultsMu["nnzs"]

    slsLbda = resultsLbda["sls"]
    slsMu = resultsMu["sls"]

    slspLbda = slsp["slspLbda"]
    slspMu = slsp["slspMu"]

    paccLbda = plot(lambdas, train_accuraciesLbda, xscale=:log10, xticks=xticks, yticks=yticks2, label="training accuracy", ls=:auto, ylabel="%", xlabel="τ", leg=:bottom, dpi=600)
    plot!(lambdas, test_accuraciesLbda, label="test accuracy", ls=:auto)
    plot!(lambdas, slsLbda, label="T-I measure", ls=:solid)
    plot!(lambdas, slspLbda, label="T-I meas. (incl. # zeros)", ls=:dash)
    savefig(paccLbda, filedir * "figures/"*problem_name*"accuracyLbda.pdf")
    paccMu = plot(mus, train_accuraciesMu, xscale=:log10, xticks=xticks, yticks=yticks2, label="training accuracy", ls=:auto, ylabel="%", xlabel="μ", leg=:bottom, dpi=600)
    plot!(mus, test_accuraciesMu, label="test accuracy", ls=:auto)
    plot!(mus, slsMu, label="T-I measure", ls=:solid)
    plot!(mus, slspMu, xscale=:log10, xticks=xticks, label="T-I meas. (incl. # zeros)", ls=:dash)
    savefig(paccMu, filedir * "figures/"*problem_name*"accuracyMu.pdf")

    ################################################

    n0 = 28*28
    ny = 10

    n1 = 512
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    ax1 = axs[1].twinx()
    
    ax1.tick_params(axis="y", labelcolor="darkred")
    # ax1.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax1.set_yscale("linear")
    axs[1].tick_params(axis="y", labelcolor="black")
    axs[1].set_xlabel("μ")
    axs[1].set_ylabel("test loss", color="black", fontsize=20)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    ax2 = axs[2].twinx()
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax2.set_yscale("linear")
    axs[2].tick_params(axis="y", labelcolor="black")
    axs[2].set_xlabel("τ")
    # axs[2].set_ylabel("test loss", color="black", fontsize=20)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")

    axs[1].plot(mus, test_lossesMu, "black")
    ax1.plot(mus, (((n0*n1)+(n1*ny)) .- nnzsMu), "darkred")

    axs[2].plot(lambdas, test_lossesLbda, "black")
    ax2.plot(lambdas, (((n0*n1)+(n1*ny)) .- nnzsLbda), "darkred")
    
    fig.tight_layout()
    PyPlot.savefig(filedir * "figures/"*problem_name*"loss_combinedLbdaMu.pdf", dpi=600)
end

function PlotResultsAny(problem_names)
    xticks = [1e7,1e6,1e5,1e4,1e3,1e2,1e1,1e0]
    yticks = [1e7,1e6,1e5,1e4,1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
    yticks2 = [10,30,50,70,90,100]

    ptes = []
    paccs = []
    ptimes = []

    leg_pte = plot(zeros(1,2), showaxis = false, grid = false, label = ["GD" "GGN-SCORE"], ls=[:dashdot :dashdot], legend_columns=-1, legendfontsize=8, leg=:bottom)
    push!(ptes, leg_pte)
    leg_pacc = plot(zeros(1,4), showaxis = false, grid = false, label = ["GD (test)" "GGN-SCORE (test)" "GD (train)" "GGN-SCORE (train)"], ls=[:dashdot :dashdot :solid :solid], legend_columns=-1, legendfontsize=8, leg=:bottom)
    push!(paccs, leg_pacc)
    leg_ptime = plot(zeros(1,4), showaxis = false, grid = false, label = ["GD (train)" "GGN-SCORE (train)" "GD (test)" "GGN-SCORE (test)"], ls=[:solid :solid :dashdot :dashdot], legend_columns=-1, legendfontsize=8, leg=:bottom)
    push!(ptimes, leg_ptime)

    for problem_name in problem_names
        activation="relu"
        batch_size=64
        n1=128
        if problem_name == "letter"
            batch_size = 8
        end
        res_gd = load(filedir*"tmp/single_$(problem_name)_"*activation*"_gd_$(n1)_$(batch_size).jld")
        res_ggn = load(filedir*"tmp/single_$(problem_name)_"*activation*"_ggn_$(n1)_$(batch_size).jld")

        pte = plot([res_gd["test_losses"], res_ggn["test_losses"]], yscale=:log10, xscale=:log10, xticks=xticks, yticks=yticks, label=["GD" "GGN-SCORE"], ls=[:dashdot :dashdot], xlabel="iteration number", leg=false, title=problem_name, dpi=600)
        push!(ptes, pte)

        pacc = plot([res_gd["test_accuracies"], res_ggn["test_accuracies"]], xscale=:log10, xticks=xticks, yticks=yticks2, label=["GD (test)" "GGN-SCORE (test)"], ls=[:dashdot :dashdot], xlabel="iteration number", leg=false, title=problem_name, dpi=600)
        plot!([res_gd["train_accuracies"], res_ggn["train_accuracies"]], label=["GD (train)" "GGN-SCORE (train)"])
        push!(paccs, pacc)

        ptime = plot([max.(res_gd["times"], 1e-9), max.(res_ggn["times"], 1e-9), max.(res_gd["times"], 1e-9), max.(res_ggn["times"], 1e-9)], [res_gd["train_accuracies"], res_ggn["train_accuracies"], res_gd["test_accuracies"], res_ggn["test_accuracies"]], yticks=yticks2, label=["GD (train)" "GGN-SCORE (train)" "GD (test)" "GGN-SCORE (test)"], ls=[:solid :solid :dashdot :dashdot], xlabel="time [s]", leg=false, title=problem_name, dpi=600)
        push!(ptimes, ptime)
        
    end

    pte_all = plot(ptes..., ylabel=reshape(vcat("", "test loss", ["" for i in 3:length(ptes)]), (1,:)), layout=@layout([[_ d{.2h} _];[a b c]]), left_margin=[12mm 12mm 12mm], size=(1700,550), sharex = true, dpi=600)
    savefig(pte_all, filedir * "figures/real_testloss_all.pdf")

    pacc_all = plot(paccs..., ylabel=reshape(vcat("", "accuracy [%]", ["" for i in 3:length(paccs)]), (1,:)), layout=@layout([[_ d{.2h} _];[a b c]]), left_margin=[12mm 12mm 12mm], size=(1700,550), dpi=600)
    savefig(pacc_all, filedir * "figures/real_testacc_all.pdf")

    ptime_all = plot(ptimes..., ylabel=reshape(vcat("", "accuracy [%]", ["" for i in 3:length(ptimes)]), (1,:)), layout=@layout([[_ d{.2h} _];[a b c]]), left_margin=[12mm 12mm 12mm], size=(1700,550), dpi=600)
    savefig(ptime_all, filedir * "figures/real_traintesttime_all.pdf")
end

function load_plot_results_lbda_mu(problem_name)
    resultsLbda = load(filedir*"tmp/lbda_"*problem_name*"_relu_ggn_512_16.jld")
    resultsMu = load(filedir*"tmp/mu_"*problem_name*"_relu_ggn_512_16.jld")
    if problem_name == "mnist"
        resultsLbdaTS = load(filedir*"tmp/lbda_"*problem_name*"-t-s_silu_ggn_1024_16.jld")
        resultsMuTS = load(filedir*"tmp/mu_"*problem_name*"-t-s_silu_ggn_1024_16.jld")
        slsp = load(filedir*"tmp/slsp_"*problem_name*"_relu_ggn_512_16.jld")
        slspTS = load(filedir*"tmp/slsp_"*problem_name*"-t-s_silu_ggn_1024_16.jld")
        # save_act_stability(resultsLbda, resultsLbdaTS, resultsMu, resultsMuTS)
        plotResultsLbdaMu(problem_name, resultsLbda, resultsLbdaTS, resultsMu, resultsMuTS, slsp, slspTS)
    else
        slsp = Dict()
        slsp["slspLbda"] = [res[end] for res in resultsLbda["slps"]]
        slsp["slspMu"] = [res[end] for res in resultsMu["slps"]]
        plotResultsLbdaMu(problem_name, resultsLbda, resultsMu, slsp)
    end
end


## NOTE: must run an experiment and save the results before loading

### to run an experiment:
#### to save results, set save_results=true
#### choose problem_name from ["pendigits", "letter", "avila"]
#### algorithms from ["ggn", "gd"]
RunExpAny(;algo="ggn", problem_name="pendigits", n1=128, batch_size=8, save_results=false)

### for MNIST and FashionMNIST experiments:
#### for mnist teacher-student experiment, set problem_name="mnist-t-s", and set the parameters as in the paper
#### choose problem_name from ["mnist", "fashionmnist", "mnist-t-s"]
# RunMNISTExpLambda(;algo="ggn", problem_name="mnist", activation="relu", n1=512, n1_star=16, batch_size=16, max_epoch=1, save_results=false);
# RunMNISTExpMu(;algo="ggn", problem_name="mnist", activation="relu", n1=512, n1_star=16, batch_size=16, max_epoch=1, save_results=false);

## LOADING and PLOTTING:
### load and plot mnist and fashionmnist results:
# load_plot_results_lbda_mu("mnist")

### load and plot UCI datasets results:
# PlotResultsAny(["pendigits", "letter", "avila"])



