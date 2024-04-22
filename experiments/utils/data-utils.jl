"""
    Load data utility functions for:
        Regularized Gauss-Newton for learning overparameterized neural networks.
"""

using MLUtils
using MLDatasets
using DelimitedFiles
using LIBSVMdata
using Flux
using Flux: onehotbatch, onecold
using Random
using Distributions
function rr()
    rng = MersenneTwister(1234)
    return rng
end

Random.seed!(1234)

function permute_inp_data(inp, permutation)
    return inp[permutation]
end
function create_permuted_datasets(Xtrain, Xtest, permutation)
    permuted_Xtrain = [permute_inp_data(x, permutation) for x in eachslice(Xtrain, dims=ndims(Xtrain))]
    permuted_Xtest = [permute_inp_data(x, permutation) for x in eachslice(Xtest, dims=ndims(Xtest))]
    permuted_Xtrain = reshape(reduce(hcat,permuted_Xtrain), size(Xtrain))
    permuted_Xtest = reshape(reduce(hcat,permuted_Xtest), size(Xtest))
    return permuted_Xtrain, permuted_Xtest
end
function get_data(;data_name="mnist", t_s=false, perm_mode=false, shuff_mode=false, seq_mode=false)
    datadir = pwd()*"/experiments/data/"
    if data_name in ["mnist", "fashionmnist"]
        if data_name == "mnist"
            train_dataset = MNIST(:train)
            test_dataset = MNIST(:test)
        else data_name == "fashionmnist"
            train_dataset = FashionMNIST(:train)
            test_dataset = FashionMNIST(:test)
        end
        train_inps, train_labels = train_dataset[:]
        test_inps, test_labels = test_dataset[:]
    elseif data_name in ["avila", "pendigits", "letter"]
        if data_name == "avila"
            data_tr = readdlm(datadir*"avila/avila-tr.txt", ',')
            data_te = readdlm(datadir*"avila/avila-ts.txt", ',')
            train_inps, train_labels = Matrix(data_tr[:,1:end-1]'), data_tr[:,end]
            test_inps, test_labels = Matrix(data_te[:,1:end-1]'), data_te[:,end]
        else
            if data_name == "letter"
                Xtr, train_labels = LIBSVMdata.load_dataset("letter.scale.tr", dense=true, replace=false, verbose=false)
                Xte, test_labels = LIBSVMdata.load_dataset("letter.scale.t", dense=true, replace=false, verbose=false)
            else
                Xtr, train_labels = LIBSVMdata.load_dataset("pendigits", dense=true, replace=false, verbose=false)
                Xte, test_labels = LIBSVMdata.load_dataset("pendigits.t", dense=true, replace=false, verbose=false)
            end
            train_inps = Matrix(Xtr')
            test_inps = Matrix(Xte')
        end
    end
    if perm_mode
        pixel_indices = shuffle(rr(), 1:prod(size(train_inps)[1:end-1]))
        train_inps, test_inps = create_permuted_datasets(train_inps, test_inps, pixel_indices)
    end
    
    Xtrain_ = stack([x for x in eachslice(train_inps, dims=ndims(train_inps))][1:3000], dims=ndims(train_inps))
    ytrain_ = train_labels[1:3000]
    Xtrain_bal, ytrain_bal = undersample(Xtrain_, ytrain_; shuffle=false)
    Xtrain_bal = Array{Float64, ndims(Xtrain_)}(Xtrain_bal)
    Xtrain_bal = Flux.flatten(Xtrain_bal)
    Xtrain = Flux.flatten(train_inps)
    Xtest = Flux.flatten(test_inps)
    Xtrain_t_s = reduce(hcat, (Xtrain_bal, Xtest))
    ytrain_bal = onehotbatch(ytrain_bal, sort(unique(ytrain_bal)))
    ytrain = onehotbatch(train_labels, sort(unique(train_labels)))
    ytest = onehotbatch(test_labels, sort(unique(test_labels)))
    ytrain_t_s = reduce(hcat, (ytrain_bal, ytest))
    if t_s
        Xtrain = Xtrain_t_s
        ytrain = ytrain_t_s
    end
    if shuff_mode # optionally shuffle the training data
        shuff = randperm(rr(), size(Xtrain, 2))
        Xtrain = Xtrain[:, shuff]
        ytrain = ytrain[:, shuff]
    end
    if seq_mode
        Xtrain = [Xtrain[:,i] for i in axes(Xtrain,2)]
        Xtest = [Xtest[:,i] for i in axes(Xtest,2)]
    else
        Xtrain, ytrain, Xtest, ytest = Matrix{Float64}(Xtrain'), Matrix{Float64}(ytrain'), Matrix{Float64}(Xtest'), Matrix{Float64}(ytest')
    end

    return Xtrain, ytrain, Xtest, ytest
end

function data_loader(X, y; batchsize::Int=64, shuffle::Bool=true)
    MLUtils.DataLoader((data=X', label=y'), batchsize=batchsize, shuffle=shuffle)
end
function accuracy(model, X, y)
    ŷ = model(X)
    correct = onecold(ŷ) .== onecold(y)
    round(100 * mean(correct); digits=2)
end
function get_accuracy(model, Xtrain, ytrain)
    (x, y) = only(data_loader(Xtrain, ytrain; batchsize=size(Xtrain,1)))
    accuracy(model, x, y)
end