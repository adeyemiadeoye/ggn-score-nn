"""
    Investigate training and test loss evolution vs time:
        Regularized Gauss-Newton for learning overparameterized neural networks.
"""

include(joinpath(@__DIR__, "..", "src/NNOptimization.jl"))
include(joinpath(@__DIR__, "utils/plot-utils.jl"))
include(joinpath(@__DIR__, "utils/problem-utils.jl"))

using JLD

filedir = pwd() * "/experiments/"

m_train = 1000
m_test = 2000
n0 = 20
n1 = 500
n1_star = 5
lbda = 1e-4
κn = 1/sqrt(n1)
rootn = (κn == 1/sqrt(n1))
κW = 1
lazy = (κW < 1)
μ = 1/κn
n_epochs = 4000

function RUNExpTrainTestLoss(;algo="ggn",save_results=true,lazy=lazy,rootn=rootn)

    if algo == "ggn"
        if lazy
            α = 1/n0
        else
            α = 0.95
        end
    else
        α = 1.
    end
    λ = lbda
    max_epoch = n_epochs
    if algo == "gd"
        λ = 0
        max_epoch = 10000
    end
    (_, testdata, model, optimizer, reg_name, hμ) = gen_problem_data(problem_name="syn", m_train=m_train, m_test=m_test, n0=n0, n1=n1, n1_star=n1_star, λ=λ, μ=μ, κn=κn, κW=κW, algo=algo)

    result = NNOptimization.optimize_model!(model, optimizer, testdata, reg_name, hμ, n1; α=α, n_epoch=max_epoch, f_tol=1e-8, verbose=1)
    
    if save_results
        if λ == 0
            save(filedir*"tmp/$(algo)_$(m_train)_$(m_test)_$((n0*n1)+n1)_no_reg.jld", "result", result)
        else
            if lazy
                rootn == true ? appn = "rootn" : appn = "n"
                save(filedir*"tmp/$(algo)_$(m_train)_$(m_test)_$((n0*n1)+n1)_lazy_$appn.jld", "result", result)
            else
                save(filedir*"tmp/$(algo)_$(m_train)_$(m_test)_$((n0*n1)+n1).jld", "result", result)
            end
        end
    end
    return result
end

function plotTrainTestLoss(res_gd, res_ggn)
    yticks = [1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]

    pte = plot([res_gd.Rte, res_ggn.Rte], yscale=:log10, xscale=:log10, yticks=yticks, label=["GD" "GGN-SCORE"], ls=[:dashdot :dashdot], ylabel="test loss", xlabel="iteration number", leg=:topright, dpi=600)
    savefig(pte, filedir * "figures/testloss_$(m_train)_$(m_test)_$((n0*n1)+n1).pdf")

    ptime = plot([max.(res_gd.times, 1e-9), max.(res_ggn.times, 1e-9), max.(res_gd.times, 1e-9), max.(res_ggn.times, 1e-9)], [res_gd.Rtr, res_ggn.Rtr, res_gd.Rte, res_ggn.Rte], yscale=:log10, yticks=yticks, label=["GD (train)" "GGN-SCORE (train)" "GD (test)" "GGN-SCORE (test)"], ls=[:solid :solid :dashdot :dashdot], ylabel="loss", xlabel="time [s]", leg=:right, dpi=600)
    savefig(ptime, filedir * "figures/traintesttime_$(m_train)_$(m_test)_$((n0*n1)+n1).pdf")
end


## NOTE: must run an experiment and save the results before loading

### run experiment
res = RUNExpTrainTestLoss(algo="ggn",save_results=false)

### to load and plot the results:
# res_gd = load(filedir*"tmp/gd_$(m_train)_$(m_test)_$((n0*n1)+n1)_no_reg.jld")["result"]
# res_ggn = load(filedir*"tmp/ggn_$(m_train)_$(m_test)_$((n0*n1)+n1).jld")["result"]
# plotTrainTestLoss(res_gd, res_ggn)


;