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

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")


"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
# Optimize Function:
function optimize(f, g, x0, n, prob)
    if prob == "simple1"
        x_H, f_H = newstrov_momentum(f, g, x0, n, 0.9, 0.5)
    elseif prob == "simple2"
        x_H, f_H = newstrov_momentum(f, g, x0, n, 2, 0.5)
        #x_H, f_H = G_Descent(f, g, x0, 40, 0.018)
    elseif prob == "simple3"
        x_H, f_H = newstrov_momentum(f, g, x0, n, 2, 0)
        # x_H, f_H = G_Descent(f, g, x0, 100, 0.001)
    elseif prob == "secret1"
        x_H, f_H = newstrov_momentum(f, g, x0, n, 1, 0.1)# 0.1 # 99.6
    else 
        x_H, f_H = newstrov_momentum(f, g, x0, n, 1, 0.6)# 0.5 # 83 # updating step size for both

    end
    x_best = x_H[argmin(f_H)]
    return x_best
end
# Backtracking Line Search Function:
function backtracking_line_search(f, g, x0; p = 0.4, β=0.0001)
    a = g(x0)
    b = f(x0)
    d = -a/norm(a)
    α = 0.018
    while f(x0 + α*d) > b + β*α*(a⋅d)
        α *= p
    end
    return α
end
# Gradient Decent:
function G_Descent(f, g, x0, n, alpha)
    x_H = [x0]
    f_H = [f(x0)]
    # Setting Counter:
    while count(f, g) < n
        x_new = x_H[end]-alpha*g(x_H[end])
        push!(x_H, x_new)
        push!(f_H, f(x_new))
    end
    return x_H, f_H
end 
# Nestrov Momentum function:
function newstrov_momentum(f, g, x0, n, α, beta )
    v_H = 0*[x0]
    x_H = [x0]
    f_H = [f(x0)]
    while count(f,g) <= n-3
        α *= 0.85
        gradient = g(x_H[end] + beta * v_H[end])
        d = -gradient./norm(gradient)
        v_new = beta * v_H[end] + α * d
        x_new = x_H[end] + v_new
        push!(v_H, v_new)
        push!(x_H, x_new)
        push!(f_H, f(x_new))
    end
    return x_H, f_H
end       