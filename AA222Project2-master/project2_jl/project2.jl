#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
using LinearAlgebra 
using Statistics
using Random
using Distributions

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")
include("helpers.jl")


"""
    optimize(f, g, c, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `c`: Constraint function for 'f'
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, c, x, n, prob)
    if prob == "simple1"
        # ρ = 500
        # mod_f1(x) = f(x) + ρ*sum(x -> x > 0 ? x : 0, c(x))
        ρ = 9999999999^30
        mod_f1(x) = simple1(x) + ρ*sum(max.(0,c(x)).^2)
        x_H, f_H = hooke_jeeves(mod_f1, x, g, c, n, 0.1, 0.50)
        # x_H, f_H = penalty_method(f, g, c, x, n, 30, 20, 5, 100, 6)
    elseif prob == "simple2"
        ρ = 500
        mod_f2(x) = f(x) + ρ*sum(x -> x > 0 ? x : 0, c(x))
        x_H, f_H = hooke_jeeves(mod_f2, x, g, c, n, 0.1, 0.5 )
    elseif prob == "simple3"
        ρ = 500
        mod_f3(x) = f(x) + ρ*sum(x -> x > 0 ? x : 0, c(x))
        x_H, f_H = hooke_jeeves(mod_f3, x, g, c, n, 0.1, 0.5)
    elseif prob == "secret2"
        x_H, f_H = penalty_method(f, g, c, x, n, 500, 20, 90, 50, 10)
    else
        x_H, f_H = penalty_method(f, g, c, x, n, 800, 5, 2000, 40, 2)
    end
    x_best = x_H[argmin(f_H)]
    return x_best
end

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1 : n]
function hooke_jeeves(f, x, g, c, n, α, γ)
    n_dim = length(x)
    y = f(x)
    x_H = [x]
    f_H = [f(x)]
    while count(f,g,c) < (n/2-n_dim)
        improved = false
        x_best = x
        y_best = y
        for i in 1:n_dim
            for sgn in (-1,1)
                x′ = x + sgn*α*basis(i,n_dim)
                y′ = f(x′)
                if y′ < y_best
                    x_best, y_best, improved = x′, y′, true
                end
            end
        end
        x, y = x_best, y_best
        
        if !improved
            α *= γ
        end
        push!(x_H, x)
        push!(f_H, y)
    end
    return x_H, f_H
end

# Define the penalty_method function
function penalty_method(f, g, c, x0, n, ρ, γ, sigma_factor, m, m_elite)
    x_H = [x0]
    f_H = [f(x0)]
    while count(f, g, c) < n-200
        μ = x_H[end]
        Σ = Diagonal(ones(length(x_H[end])))*sigma_factor
        P = MvNormal(μ, Σ)
        mod_f(x) = f(x) + ρ*sum(x -> x > 0 ? x : 0, c(x))
        x_new = cross_entropy_method(mod_f, g, c, n, P, m, m_elite)
        
        ρ *= γ
        if c(x_new) == 0
            push!(x_H, x_new)
            push!(f_H, f(x_new))
            break
        end
        push!(x_H, x_new)
        push!(f_H, f(x_new))
    end
    return x_H, f_H
end
# cross entropy:
function cross_entropy_method(f, g, c, n, P, m, m_elite)
    samples = rand(P, m)
    order = sortperm([f(samples[:,i]) for i in 1:m])
    P = fit(typeof(P), samples[:,order[1:m_elite]])
    x = mean(P)
return x
end

