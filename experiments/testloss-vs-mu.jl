"""
    Investigate test loss vs mu:
        Regularized Gauss-Newton for learning overparameterized neural networks.
"""

include(joinpath(@__DIR__, "..", "src/NNOptimization.jl"))
include(joinpath(@__DIR__, "utils/plot-utils.jl"))
include(joinpath(@__DIR__, "utils/problem-utils.jl"))

using JLD
using Random

filedir = pwd() * "/experiments/"

m_train = 500
m_test = 1000
n0 = 5
n1 = 500
n1_star = 5
mus = 10.0.^(-3:0.1:1)
n_mus = length(mus)
κn = 1/sqrt(n1)
κW = 1
λ = 1e-4
max_epoch = 1600
n_runs = 10

Random.seed!(1234)
seeds = collect(1111:10:1111+(10*n_runs)-1)
shuffle!(seeds)

function RUNExpTrainTestMu(;algo="ggn",save_results=true)

    train_losses_runs = Vector{Vector{Float64}}(undef, n_runs)
    train_losses_reg_runs = Vector{Vector{Float64}}(undef, n_runs)
    test_losses_runs = Vector{Vector{Float64}}(undef, n_runs)
    n_epochs_runs = Vector{Vector{Float64}}(undef, n_runs)
    nnzs_runs = Vector{Vector{Float64}}(undef, n_runs)

    t = 0

    for k in 1:n_runs

        seed = seeds[k]

        train_losses_mus = Vector{Float64}(undef, n_mus)
        train_losses_reg_mus = Vector{Float64}(undef, n_mus)
        test_losses_mus = Vector{Float64}(undef, n_mus)
        n_epochs_mus = Vector{Float64}(undef, n_mus)
        nnzs_mus = Vector{Float64}(undef, n_mus)

        for i in axes(mus,1)

            print("-")

            μ = mus[i]

            (_, testdata, model, optimizer, reg_name, hμ) = gen_problem_data(problem_name="syn", m_train=m_train, m_test=m_test, n0=n0, n1=n1, n1_star=n1_star, λ=λ, μ=μ, κn=κn, κW=κW, algo=algo, seed=seed)
        
            result = NNOptimization.optimize_model!(model, optimizer, testdata, reg_name, hμ, n1; α=0.95, n_epoch=max_epoch, f_tol=1e-8, verbose=0)

            t += result.times[end]

            train_losses_mus[i] = result.Rtr[end]
            train_losses_reg_mus[i] = result.Loss[end]
            test_losses_mus[i] = result.Rte[end]
            n_epochs_mus[i] = result.iters
            nnzs_mus[i] = cardcal(vec(result.traj[:,:,end]),0.999)

        end

        print("[$k/$n_runs run(s) completed]\n")

        train_losses_runs[k] = train_losses_mus
        train_losses_reg_runs[k] = train_losses_reg_mus
        test_losses_runs[k] = test_losses_mus
        n_epochs_runs[k] = n_epochs_mus
        nnzs_runs[k] = nnzs_mus

    end

    train_losses_avg = mean(train_losses_runs)
    train_losses_reg_avg = mean(train_losses_reg_runs)
    test_losses_avg = mean(test_losses_runs)
    n_epochs_avg = mean(n_epochs_runs)
    nnzs_avg = mean(nnzs_runs)

    if save_results
        save(filedir*"tmp/mu_$(algo)_$(m_train)_$(m_test)_$((n0*n1)+n1).jld", "times", t, "mus", mus, "train_losses_avg", train_losses_avg, "train_losses_reg_avg", train_losses_reg_avg, "test_losses_avg", test_losses_avg, "n_epochs_avg", n_epochs_avg, "nnzs_avg", nnzs_avg)
    end

end

function plotTrainTestMu(results)

    mus = results["mus"]
    test_losses_avg = results["test_losses_avg"]
    nnzs_avg = results["nnzs_avg"]

    fig, axs = plt.subplots(figsize=(7, 4))
    ax1 = axs.twinx()

    axs.plot(mus, test_losses_avg, "black")
    axs.tick_params(axis="y", labelcolor="black")
    axs.set_xlabel("μ")
    axs.set_ylabel("test loss", color="black", fontsize=20)
    axs.set_xscale("log")
    axs.set_yscale("log")
    ax1.plot(mus, (((n0*n1)+n1) .- nnzs_avg), "darkred")
    ax1.tick_params(axis="y", labelcolor="darkred")
    # ax1.set_ylabel("# zeros", color="darkred", fontsize=20)
    ax1.set_yscale("linear")

    fig.tight_layout()
    PyPlot.savefig(filedir * "figures/lossnnzmu_combined_$(m_train)_$(m_test)_$((n0*n1)+n1).pdf", dpi=600)

end


## NOTE: must run an experiment and save the results before loading

### to run the experiment:
#### to save results, set save_results=true
RUNExpTrainTestMu(save_results=false)

# results = load(filedir*"tmp/mu_ggn_$(m_train)_$(m_test)_$((n0*n1)+n1).jld")
# plotTrainTestMu(results)

;