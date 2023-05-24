using Plots
using LinearAlgebra
include("helpers.jl")
include("simple.jl")

# Simple 1:
function simple1_init()
    return rand(2) * 2.0
end
# Function:
function simple1(x,y)
    return -x * y + 2.0 / (3.0 * sqrt(3.0))
end
# Gradient:
function simple1_gradient(x::Vector)
    return [-x[2], -x[1]]
end
# Constraints:
function simple1_constraints(x::Vector)
    return [x[1] + x[2]^2 - 1,
            -x[1] - x[2]]
end
# Constraint 1:
function simple1_constraints1(x,y)
    return x + y^2 - 1
end
# Constraint 2:
function simple1_constraints2(x,y)
    return -x - y
end

# Simple 2:
function simple2_init()
    return rand(2) .* 2.0 .- 1.0
end
# Function:
function simple2(x,y)
    return (1.0 - x)^2 + 100.0 * (y - x^2)^2
end
# Gradient:
function simple2_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    return storage
end
# Constraints:
function simple2_constraints(x::Vector)
    return [(x[1]-1)^3 - x[2] + 1,
            x[1] + x[2] - 2]
end
# Constraint 1:
function simple2_constraints1(x, y)
    return (x-1)^3 - y + 1
end
# Constraint 2:
function simple2_constraints2(x, y)
    return x + y - 2
end


# Simple 3:
function simple3_init()
    b = 2.0 .* [1.0, -1.0, 0.0]
    a = -2.0 .* [1.0, -1.0, 0.0]
    return rand(3) .* (b-a) + a
end
# function:
function simple3(x, y, z)
    return x - 2*y + z + sqrt(6.0)
end

function simple3_gradient(x::Vector)
    return [1, -2, 1]
end

function simple3_constraints(x::Vector)
    return [x[1]^2 + x[2]^2 + x[3]^2 - 1]
end


# Optimization Function:
basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1 : n]
function hooke_jeeves(f, g, c, x, n, α, γ)
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

# Executing Mixed Penalty:
ρ = 10
mod_f1(x) = simple1(x) + ρ*sum(x -> x > 0 ? x : 0, simple1_constraints(x))
#mod_f2(x) = simple1(x) + ρ*sum(x -> x > 0 ? x : 0, simple2_constraints(x))
#xhistory, fhistory =hooke_jeeves2(mod_f, [-1.5, -1.5], simple1_gradient, simple1_constraints, 2000, 0.1, 0.5 )

# Plotting for Simple 1:
# Contour for Simple 1:
xr = -3:0.1:3
yr = -3:0.1:3
contour(xr, yr, simple1, levels = LinRange(-5,10,21), colorbar = false, c = cgrad(:viridis, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 1", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
contour!(xr, yr, simple1_constraints1, levels = [0], colorbar = false, c = cgrad(:plasma, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 1", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
contour!(xr, yr, simple1_constraints2, levels = [0], colorbar = false, c = cgrad(:plasma, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 1", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
# Start Point 1:
x_H1, f_H1 = hooke_jeeves(mod_f1, simple1_gradient, simple1_constraints, [-1.5,-1.5], 2000, .1, 0.5)
plot!([x_H1[i][1] for i =1:length(x_H1)], [x_H1[i][2] for i =1:length(x_H1)], color = :black, label = "Path 1 [-1.0,-1.0]")
# Start Point 2:
# x_H2, f_H2 = hookjeeves(simple1, simple1_gradient, simple1_constraints, [0.5,0.5], 10000, .0001, 0.4)
# plot!([x_H2[i][1] for i =1:length(x_H2)], [x_H2[i][2] for i =1:length(x_H2)], color = :blue, label = "Path 2 [0.5,0.5]")
# # Start Point 3:
# x_H3, f_H3 = hookjeeves(simple1, simple1_gradient, simple1_constraints, [1.5, 1.5], 20000, .0001, 0.4 )
# plot!([x_H3[i][1] for i =1:length(x_H3)], [x_H3[i][2] for i =1:length(x_H3)], color = :red, label = "Path 3 [1.5, 1.5]")
# plot!(legend=:outerbottom, legendcolumns=3)
title!("Simple1 Contour")
savefig("Simple1_contour.png")

# # Plotting for Simple 2:
# Contour for Simple 2:
# xr = -3:0.1:3
# yr = -3:0.1:3
# contourf(xr, yr, simple2, levels=[10,25,50,100,200,250,300], colorbar = false, c = cgrad(:viridis, rev=true), legend = true, 
#         xlims = (-2, 2), ylims = (-2, 2), xlabel = "x1", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
# contour!(xr, yr, simple2_constraints1, levels=[0], colorbar = false, c = cgrad(:viridis, rev=true), legend = true, 
#         xlims = (-2, 2), ylims = (-2, 2), xlabel = "x1", ylabel = "x2", color = :red, aspectratio = :equal, clim = (-10,10))
# contour!(xr, yr, simple2_constraints2, levels=[0], colorbar = false, c = cgrad(:viridis, rev=true), legend = true, 
#         xlims = (-2, 2), ylims = (-2, 2), xlabel = "x1", ylabel = "x2", color = :orange, aspectratio = :equal, clim = (-10,10))
# # Start Point 1:
# x_H1, f_H1 = hook_jeeves(mod_f2, simple2_gradient, simple2_constraints, [-1.5,-1.5], 2000, .0001, 0.4)
# plot!([x_H1[i][1] for i =1:length(x_H1)], [x_H1[i][2] for i =1:length(x_H1)], color = :black, label = "Path 1 [-1.0,-1.0]")
# # # Start Point 2:
# # x_H2, f_H2 = penalty_method(simple2, simple2_gradient, simple2_constraints, [0.5,0.5], 4000, 18.5, 0.25, .5, 200)
# #plot!([x_H2[i][1] for i =1:length(x_H2)], [x_H2[i][2] for i =1:length(x_H2)], color = :blue, label = "Path 2 [0.5,0.5]")
# # # Start Point 3:
# #x_H3, f_H3 = penalty_method(simple2, simple2_gradient, simple2_constraints, [1.5, 1.5], 6000, 18.5, 0.25, .5, 200)
# #plot!([x_H3[i][1] for i =1:length(x_H3)], [x_H3[i][2] for i =1:length(x_H3)], color = :red, label = "Path 3 [1.5, 1.5]")
# # plot!(legend=:outerbottom, legendcolumns=3)
# title!("Simple2 Contour")
# savefig("Simple2_contour.png")
