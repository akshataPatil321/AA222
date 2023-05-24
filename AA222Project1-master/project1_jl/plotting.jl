using Plots
using LinearAlgebra
include("helpers.jl")

# Rosenbrock Function:
function rosenbrock_init()
    return clamp.(randn(2), -3.0, 3.0)
end
# Rosenbrock Function: 2D
function rosenbrock(x, y)
    return (1.0 - x)^2 + 100.0 * (y - x ^2)^2
end
# Rosenbrock Gradient:
function rosenbrock_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    return storage
end

# Himmelblau's Function:
function himmelblau_init()
    return clamp.(randn(2), -3.0, 3.0)
end
# Function:
function himmelblau(x,y)
    return (x^2 + y - 11)^2 + (x + y^2 - 7)^2
end
# Gradient:
function himmelblau_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    storage[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    return storage
end

# Powell's function:
function powell_init()
    return clamp.(randn(4), -3.0, 3.0)
end
# Function:
function powell(w, x, y, z)
    return (w + 10.0 * x)^2 + 5.0 * (y- z)^2 +
        (x - 2.0 * y)^4 + 10.0 * (w - z)^4
end
# Gradient:
function powell_gradient(x::Vector)
    storage = zeros(4)
    storage[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
    storage[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
    storage[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
    storage[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
    return storage
end

# Gradient Descent:
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
# Backtracking Line Search:
function backtracking_line_search(f ,g, x0; p = 0.4, β=0.0001)
    a = g(x0)
    b = f(x0)
    d = -a/norm(a)
    α = 0.002
    while f(x0 + α*d) > b + β*α*(a⋅d)
        α *= p
    end
    return α
end
# Nestrov Momentum:
#function newstrov_momentum(f, g, x0, n, α, beta )
function newstrov_momentum(f::Function, g::Function, x0::Vector, n::Int, α::Real, beta::Real)
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
# Plotting for Rosenbrock function: 
x_H1, f_H1 = newstrov_momentum(rosenbrock, rosenbrock_gradient, [-1.0,-1.0], 20, 1, 0.5 )
# Contour of Rosenbrock:ex
xr = -2:0.1:3
yr = -2:0.1:3
contour!(xr, yr, rosenbrock, levels=[10,25,50,100,250,300], colorbar = false, c = cgrad(:viridis, rev=true), legend = false, 
        xlims = (-2, 2), ylims = (-2, 2), xlabel = "x1", ylabel = "x2", aspectratio = :equal, clim = (2,500))
plot!([x_H1[i][1] for i =1:length(x_H1)], [x_H1[i][2] for i =1:length(x_H1)], color = :black, label = "Path 1 [-1.0,-1.0]")
x_H2, f_H2 = newstrov_momentum(rosenbrock, rosenbrock_gradient, [0.5, 0.5], 40, 1, 0.5 )
plot!([x_H2[i][1] for i =1:length(x_H2)], [x_H2[i][2] for i =1:length(x_H2)], color = :blue, label = "Path 2 [0.5, 0.5]")
x_H3, f_H3 = newstrov_momentum(rosenbrock, rosenbrock_gradient, [1.5, 1.5], 60, 1, 0.5 )
plot!([x_H3[i][1] for i =1:length(x_H3)], [x_H3[i][2] for i =1:length(x_H3)], color = :red, label = "Path 3 [1.5, 1.5]")
plot!(legend=:outerbottom, legendcolumns=3)
title!("Rosenbrock Contour")
savefig("Rosenbrock_contour.png")
# Convergence Plot:
plot(collect(1:length(f_H1)), f_H1, xlabel = "Iteration", ylabel = "f(x)", color = :black, label = "Path 1 [-1.0,-1.0]")
plot!(collect(1:length(f_H2)), f_H2, xlabel = "Iteration", ylabel = "f(x)", color = :blue, label = "Path 2 [0.5, 0.5]")
plot!(collect(1:length(f_H3)), f_H3, xlabel = "Iteration", ylabel = "f(x)", color = :red, label = "Path 3 [1.5, 1.5]")
plot!( legend=:topright, legendcolumns=1)
title!("Rosenbrock Function Convergence:")
savefig("Rosenbrock_convergence.png")

# Plotting for Himmelblau's function:
# x_H, f_H = G_Descent(himmelblau, himmelblau_gradient, [-1.0,-1.0], 40.0, 0.015)
x_H, f_H = newstrov_momentum(himmelblau, himmelblau_gradient, [-1.0,-1.0], 40, 2.0, 0.5)
x_H1, f_H1 = newstrov_momentum(himmelblau, himmelblau_gradient, [0.5, 0.5], 60, 2.0, 0.5)
x_H2, f_H2 = newstrov_momentum(himmelblau, himmelblau_gradient, [1.5, 1.5], 80, 2.0, 0.5)
# Convergence of Himmelblau's:ex
plot(collect(1:length(f_H)), f_H, xlabel = "Iteration", ylabel = "f(x)", color = :black, label = "Path 1 [-1.0,-1.0]")
plot!(collect(1:length(f_H1)), f_H1, xlabel = "Iteration", ylabel = "f(x)", color = :blue, label = "Path 2 [0.5, 0.5]")
plot!(collect(1:length(f_H2)), f_H2, xlabel = "Iteration", ylabel = "f(x)", color = :red, label = "Path 3 [1.5, 1.5]")
plot!( legend=:topright, legendcolumns=1)
title!("Himmelblau's Function Convergence:")
savefig("Himmelblau_convergence.png")

# PLotting for Powell's function:
x_H, f_H = newstrov_momentum(powell, powell_gradient, [-1.0,-1.0, -1.0, -1.0], 100, 1, 0 )
x_H1, f_H1 = newstrov_momentum(powell, powell_gradient, [0.5, 0.5, 0.5, 0.5], 120, 1, 0 )
x_H2, f_H2 = newstrov_momentum(powell, powell_gradient, [1.5, 1.5, 1.5, 1.5], 140, 1, 0 )
# Convergence Plot:
plot(collect(1:length(f_H)), f_H, xlabel = "Iteration", ylabel = "f(x)", color = :black, label = "Path 1 [-1.0,-1.0, -1.0, -1.0]")
plot!(collect(1:length(f_H1)), f_H1, xlabel = "Iteration", ylabel = "f(x)", color = :blue, label = "Path 2 [0.5, 0.5, 0.5, 0.5]")
plot!(collect(1:length(f_H2)), f_H2, xlabel = "Iteration", ylabel = "f(x)", color = :red, label = "Path 3 [1.5, 1.5, 1.5, 1.5]")
plot!( legend=:topright, legendcolumns=1)
title!("Powell's Function Convergence:")
savefig("Powell_convergence.png")