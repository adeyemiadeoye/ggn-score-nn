"""
    Problem utility functions for:
        Regularized Gauss-Newton for learning overparameterized neural networks.
"""

include(joinpath(@__DIR__, "grad-utils.jl"))
include(joinpath(@__DIR__, "data-utils.jl"))

using SelfConcordantSmoothOptimization
using BSON
using BSON: @save
using Random

function gen_problem_data(;
    problem_name="syn", m_train=500, m_test=1000, n0=100, n1=50, n1_star=5, reg_name="l1", λ=1e-4, μ=1.0, activation="silu", κn=1, κW=0.2, algo="ggn", use_prox=false, t_model_dir=nothing, save_t_model=true, verbose_t_s=0, max_epochs_t_s=250, seed=1234)

    """
        m_train:    number of training samples
        m_test:     number of test samples
        n0:         input dimension
        n1:         number of hidden neurons of (student network)
        n1_star:    number of hidden neurons (teacher network)
        reg_name:   regularization to use
        λ:          regularization strength parameter
        μ:          smoothing parameter
        activation: activation function
        κn:         scaling parameter (applies to student network)
        κW:         scale of initial weight
        algo:       optimizer to use
        use_prox:   to use a proximal step or not
        seed:       seed number
    """

    Random.seed!(seed)

    if activation == "silu"
        π_fn = π_star = silu
        π_fn_grad = silu_grad
    elseif activation == "relu"
        π_fn = π_star = relu
        π_fn_grad = relu_grad
    else
        Base.error("Please choose 'silu' or 'relu' activation.")
    end
    
    if problem_name == "syn"
        u_star = randn(n1_star,n0)
        u_star = u_star ./ sqrt.(sum(u_star.^2, dims=2))
        v_star  = sign.(randn(n1_star))
        Φ_star(X) = sum(v_star .* π_star.(u_star*X'), dims=1)

        Xtrain = randn(m_train, n0)
        Xtrain = Xtrain ./ sqrt.(sum(Xtrain.^2, dims=2))
        ytrain = Matrix(Φ_star(Xtrain)')
        Xtest  = randn(m_test, n0)
        Xtest  = Xtest ./ sqrt.(sum(Xtest.^2, dims=2))
        ytest  = Matrix(Φ_star(Xtest)')
        Xtest0 = nothing
        ytest0 = nothing

        # initialization
        W0 = randn(n1, n0)
        W0 = reduce(hcat, (W0, rand(n1)))
        W0 = κW*W0
        θ0 = vec(W0)
        n1n0 = n1*n0
        # define the scaled neural network
        Φu = (θ) -> reshape(θ[1:n1n0], (n1, n0))
        Φv = (θ) -> θ[n1n0+1:end]
        Φ = (X, θ) -> κn*sum(Φv(θ) .* π_fn.(Φu(θ)*X'), dims=1)'

        # get gradient fnctions
        grad_Lx, jac_Φx, grad_Ly, hess_Ly = get_derivatives_1D(n1, κn, π_fn, π_fn_grad)
    elseif problem_name in ["mnist-t-s", "mnist", "fashionmnist-t-s", "fashionmnist", "pendigits", "pendigits-t-s", "letter", "letter-t-s", "avila", "avila-t-s"]
        t_s = (problem_name in ["mnist-t-s", "fashionmnist-t-s", "avila-t-s", "letter-t-s", "pendigits-t-s"])
        if problem_name in ["mnist", "mnist-t-s"]
            data_name = "mnist"
        elseif problem_name in ["fashionmnist", "fashionmnist-t-s"]
            data_name = "fashionmnist"
        elseif problem_name in ["pendigits", "pendigits-t-s"]
            data_name = "pendigits"
        elseif problem_name in ["letter", "letter-t-s"]
            data_name = "letter"
        elseif problem_name in ["avila", "avila-t-s"]
            data_name = "avila"
        else
            Base.error("Invalid data_name.")
        end
        Xtrain, ytrain, Xtest, ytest = get_data(data_name=data_name, t_s=t_s, shuff_mode=false)

        n0 = size(Xtrain,2)
        ny = size(ytrain,2)
        n1n0 = n1*n0

        if t_s
            if t_model_dir !== nothing
                @info "now loading a teacher model from the supplied directory..."
                BSON.@load t_model_dir teacher_model
                @info "teacher model loaded."
            else
                # train a teacher model
                teacher_model = train_get_teacher_model(Xtrain, ytrain; act_fn=π_star, n1=n1_star, batch_size=64, max_epochs=max_epochs_t_s, verbose=verbose_t_s)
                if save_t_model
                    filedir = pwd()*"/experiments/"
                    @save filedir*"tmp/$(data_name)_teacher_model.bson" teacher_model
                end
            end

            # generate target data from trained teacher model
            ytrain = Matrix(teacher_model(Xtrain')')
            ytest = Matrix(teacher_model(Xtest')')
        end

        Xtest0 = Xtest
        ytest0 = ytest

        # initialize model training parameters using Flux
        nn_model = fmap(f64, Chain(Dense(n0 => n1, relu, bias=false, init=Flux.randn32(rr())), Dense(n1 => ny, identity, bias=false, init=rand32(rr()))))
        θ0, _ = Flux.destructure(nn_model)

        # define the scaled neural network
        Φu = (θ) -> reshape(θ[1:n1n0], (n1, n0))
        Φv = (θ) -> reshape(θ[n1n0+1:end], (ny, n1))
        Φ = (X, θ) -> κn*(Φv(θ) * π_fn.(Φu(θ)*X'))'

        # get gradient fnctions
        grad_Lx, jac_Φx, grad_Ly, hess_Ly = get_derivatives_mD(n1, κn, π_fn, π_fn_grad)
    else
        Base.error("problem_name not valid.")
    end

    traindata = Dict()
    traindata["X"] = Xtrain
    traindata["y"] = ytrain

    testdata = Dict()
    testdata["X"] = Xtest
    testdata["y"] = ytest

    # Robj (objective) as a function of y and yhat:
    Robj(y, yhat) = 0.5*mean_square_error(yhat, y)
    # Robj as a function of X, y, θ
    function Robj(X, y, θ)
        yhat = Φ(X, θ)
        loss = Robj(y, yhat)
        return loss
    end
    
    hμ = PHuberSmootherL1L2(μ)
    # define SelfConcordantSmoothOptimization problem
    model = Problem(Xtrain, ytrain, θ0, Robj, λ; Atest=Xtest0, ytest=ytest0, out_fn=Φ, grad_fx=grad_Lx, jac_yx=jac_Φx, grad_fy=grad_Ly, hess_fy=hess_Ly)

    # set optimizer
    if algo == "ggn"
        optimizer = ProxGGNSCORE(use_prox=use_prox)
    elseif algo == "gd"
        optimizer = ProxGradient(use_prox=use_prox)
    else
        Base.error("Please choose a valid algo: 'ggn' or 'gd'")
    end

    return traindata, testdata, model, optimizer, reg_name, hμ

end

function train_get_teacher_model(Xtrain, ytrain; act_fn=relu, n1=32, max_epochs=100, batch_size=nothing, verbose=1)
    local gs

    if batch_size === nothing
        batch_size = size(Xtrain,1)
    end

    n0 = size(Xtrain,2)
    ny = size(ytrain, 2)
    Random.seed!(1234)
    model = fmap(f64, Chain(Dense(n0 => n1, act_fn, bias=false, init=Flux.randn32(rr())), Dense(n1 => ny, identity, bias=false, init=randn32(rr())), softmax))
    train_loader = data_loader(Xtrain, ytrain, batchsize=batch_size)
    opt_state = Flux.setup(Adam(1e-3), model)
    l_split = "================\n"
    @info "now training a teacher model...\n"
    print(l_split)
    for epoch in 1:max_epochs
        loss = 0.0
        for (x, y) in train_loader
            l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
            Flux.update!(opt_state, model, gs[1])
            loss += l/length(train_loader)
        end

        if verbose > 0
            if mod(epoch, 50) == 0
                train_acc = get_accuracy(model, Xtrain, ytrain)
                @show epoch loss train_acc
                print(l_split)
            end
        end
    end
    @info "teacher model training done.\n"

    return model
end

silu(x;s=1) = x*1/(1+exp(-s*x))
silu_grad(x;s=1) = 1/(1 + exp(-s*x)) + x/(1 + exp(-s*x))^2*s*exp(-s*x)
relu_grad(x) = float.(x .> 0)

function cardcal(x, r)
    local k
    n = length(x)
    normx1 = norm(x, 1)
    absx = sort(abs.(x), rev=true)
    for i = 1:n
        if sum(absx[1:i]) >= r * normx1
            k = i
            break
        end
    end
    return k
end