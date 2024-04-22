"""
    Derivative utility functions for:
        Regularized Gauss-Newton for optimizing overparameterized neural networks.
"""

using LinearAlgebra

function get_derivatives_1D(n1, κn, act_fn, act_fn_grad)

    function grad_Ly(X, y, yhat)
        m = size(X,1)
        1/m*(yhat .- y)
    end
    function hess_Ly(X, y, yhat)
        m = size(X,1)
        return 1/m*Matrix(I, m, m)
    end
    function grad_Lx(X, y, θ)
        m, n0 = size(X)
        n1n0 = n1*n0
        u = θ[1:n1n0]
        u = reshape(u, (n1, n0))
        v = θ[n1n0+1:end]
        act = act_fn.(u*X')
        out = κn*sum(v .* act, dims=1)'
        gradLy = grad_Ly(X, y, out)
        grad_u = v .* act_fn_grad.(act)*(X .* gradLy)
        grad_v = act*gradLy
        return vec(reduce(hcat, (grad_u, grad_v)))
    end
    function jac_Φx(X, y, yhat, θ)
        m, n0 = size(X)
        n1n0 = n1*n0
        ny = size(yhat,2)
        u = θ[1:n1n0]
        u = reshape(u, (n1, n0))
        v = θ[n1n0+1:end]
        J = zeros(m, (n0*n1)+(n1*ny))
        for i in 1:m
            x = X[i,:]
            pre_act = u*x
            act = act_fn.(pre_act)
            for j in 1:n1
                for k in 1:n0
                    J[i, (j-1)*n0 + k] = v[j]*act_fn_grad(pre_act[j])*x[k]
                end
            end
            for j in 1:ny
                for k in 1:n1
                    J[i, (n0 * n1) + (j-1)*n1 + k] = act[k]
                end
            end
        end
        return J
    end

    return grad_Lx, jac_Φx, grad_Ly, hess_Ly
end

function get_derivatives_mD(n1, κn, act_fn, act_fn_grad)
    function grad_Ly(X, y, yhat)
        m = size(X,1)
        ny = size(y,2)
        1/(ny*m)*vec((yhat .- y))
    end
    function hess_Ly(X, y, yhat)
        m = size(X,1)
        ny = size(y,2)
        return 1/(ny*m)*Matrix(I, ny*m, ny*m)
    end
    function grad_Lx(X, y, θ)
        m, n0 = size(X)
        n1n0 = n1*n0
        ny = size(y,2)
        u = reshape(θ[1:n1n0], (n1, n0))
        v = reshape(θ[n1n0+1:end], (ny, n1))
        act = act_fn.(u*X')
        out = κn*(v*act)'
        gradLy = grad_Ly(X, y, out)
        gradLy = reshape(gradLy,(m,:))
        gradu = vec(κn*((v'*gradLy') .* act_fn_grad.(u*X'))*X)
        gradv = vec((κn*act*gradLy)')
        return reduce(vcat, (gradu, gradv))
    end
    function jac_Φx(X, y, yhat, θ)
        m, n0 = size(X)
        n1n0 = n1*n0
        ny = size(y,2)
        u = reshape(θ[1:n1n0], (n1, n0))
        v = reshape(θ[n1n0+1:end], (ny, n1))
        J = zeros(ny*m, (n0*n1)+(n1*ny))
        Threads.@threads for i in 1:ny
            for j in 1:m
                x = X[j,:]
                pre_act = u*x
                J[((i-1)*m)+j, 1:n1n0] .= vec(κn*(act_fn_grad.(pre_act).*v[i,:])*x')
            end
        end
        jstart = (1:m:ny*m)
        jend = (1:ny).*m
        pre_act = X*u'
        act = κn*act_fn.(pre_act)
        Threads.@threads for i in 1:n1
            for j in 1:ny
                J[jstart[j]:jend[j], (n0*n1)+((i-1)*ny)+j] .= act[:,i]
            end
        end
        return J
    end

    return grad_Lx, jac_Φx, grad_Ly, hess_Ly
end