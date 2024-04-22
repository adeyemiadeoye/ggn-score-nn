module NNOptimization

# Based on the package https://github.com/adeyemiadeoye/SelfConcordantSmoothOptimization.jl

using SelfConcordantSmoothOptimization
using LinearAlgebra
using Random
using Dates


const NNOptimizer = SelfConcordantSmoothOptimization.ProximalMethod
const NNModel = SelfConcordantSmoothOptimization.ProxModel
const step! = SelfConcordantSmoothOptimization.iter_step!
const init! = SelfConcordantSmoothOptimization.init!

struct Result{A, D, L, T, K}
    traj::A
    Rtr::D
    Rte::D
    Loss::L
    times::T
    iters::K
end

show(io::IO, r::Result) = show(io, "")

function optimize_model!(model::NNModel, optimizer::NNOptimizer, testdata::Dict{Any, Any}, reg_name, hμ, n_neurons; α=nothing, n_epoch=100, f_tol=1e-6, save_traj=true, verbose=0)

    Xtrain, ytrain = model.A, model.y
    Xtest, ytest = testdata["X"], testdata["y"]

    if α !== nothing
        model.L = 1/α
    end

    d_misfits_tr = []
    d_misfits_te = []
    Loss = []
    times = []
    epochs = 0
    R_tr(θ) = model.f(Xtrain, ytrain, θ)
    R_te(θ) = model.f(Xtest, ytest, θ)
    θ = model.x0
    θ_prev = deepcopy(θ)

    m, n0 = size(Xtrain)
    save_traj ? traj = zeros(n_neurons, n0+1, n_epoch) : traj = nothing
    n1n0 = n_neurons * n0

    # initialize optimizer
    init!(optimizer, θ)

    t0 = now()
    for epoch in 1:n_epoch
        Δtime = (now() - t0).value/1000

        if save_traj
            u = θ[1:n1n0]
            u = reshape(u, (n_neurons, n0))
            v = θ[n1n0+1:end]
            traj[:,:,epoch] = reduce(hcat, (u,v))
        end

        d_misfit_tr = R_tr(θ)
        d_misfit_te = R_te(θ)
        reg_loss = d_misfit_tr + get_reg(model, θ, reg_name)
        
        push!(times, Δtime)
        push!(d_misfits_tr, d_misfit_tr)
        push!(d_misfits_te, d_misfit_te)
        push!(Loss, reg_loss)

        if verbose > 0
            train_info = "Epoch $(epoch-1) \t R_train: $d_misfit_tr \t R_train + g: $reg_loss \t R_test: $d_misfit_te \t Time: $Δtime"
            println(train_info)
            flush(stdout)
        end

        if d_misfit_tr ≤ f_tol
            break
        end

        θ_new = step!(optimizer, model, reg_name, hμ, Xtrain, θ, θ_prev, ytrain, I, epoch)
        θ_prev = deepcopy(θ)
        θ = θ_new

        epochs += 1

    end

    return Result(traj, d_misfits_tr, d_misfits_te, Loss, times, epochs)

end

end # module NNOptimization