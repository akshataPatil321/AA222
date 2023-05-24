using Plots
using LinearAlgebra

include("helpers.jl")
include("simple.jl")

# Simple 2:
function simple2(x1, x2)
    return (1.0 - x1)^2 + 100.0 * (x2 - x1^2)^2
end
function simple2_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    return storage
end
# function simple2_constraints(x::Vector)
#     return [(x[1]-1)^3 - x[2] + 1,
#             x[1] + x[2] - 2]
# end
function simple2_constraints1(x1, x2)
    return (x1-1)^3 - x2 + 1
end
function simple2_constraints2(x1, x2)
    return x1 + x2 - 2
end
function simple2_init()
    return rand(2) .* 2.0 .- 1.0
end

# basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1 : n]
# function hooke_jeeves2(f, x, g, c, n, α, γ)
#     n_dim = length(x)
#     y = f(x)
#     x_H = [x]
#     f_H = [f(x)]
#     c1, c2 = c(x)
#     v_H = [max(c1, c2, 0)]
#     while count(f,g,c) < (n/2-n_dim)
#         improved = false
#         x_best = x
#         y_best = y
#         for i in 1:n_dim
#             for sgn in (-1,1)
#                 x′ = x + sgn*α*basis(i,n_dim)
#                 y′ = f(x′)
#                 if y′ < y_best
#                     x_best, y_best, improved = x′, y′, true
#                 end
#             end
#         end
#         x, y = x_best, y_best
#         c1, c2 = c(x)
#         v = max(max(0, c1), max(0, c2))
#         if !improved
#             α *= γ
#         end
#         push!(x_H, x)
#         push!(f_H, y)
#         push!(v_H, v)
#     end
#     return x_H, f_H, v_H
# end

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1 : n]
function hooke_jeeves2(f, x, g, c, n, α, γ)
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

# Executing Penalty Method:
ρ = 10
mod_f2(x) = simple2(x) + ρ*sum(x -> x > 0 ? x : 0, simple2_constraints(x))

# Quad Penalty
#ρ2 = 9999999999^30
#mod_f2(x) = simple2(x) + ρ2*sum(max.(0,simple2_constraints(x)).^2)

# Plotting for Simple 2:
x_H1, f_H1 = hooke_jeeves2(mod_f2, [-1.0,-1.0], simple2_gradient, simple2_constraints, 2000, 0.1, 0.5 )
x_H2, f_H2 = hooke_jeeves2(mod_f2, [2.0, 2.0], simple2_gradient, simple2_constraints, 4000, 0.1, 0.5 )
x_H3, f_H3 = hooke_jeeves2(mod_f2, [.5,.5], simple2_gradient, simple2_constraints, 6000, 0.1, 0.5 )

xr = -3:0.1:3
yr = -3:0.1:3
contourf(xr, yr, simple2, levels=[10,25,50,100,250,300], colorbar = false, c = cgrad(:viridis, rev = true), legend = false,
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 2", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
contour!(xr, yr, simple2_constraints1, levels = [0], colorbar = false, c = cgrad(:plasma, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 2", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
contour!(xr, yr, simple2_constraints2, levels = [0], colorbar = false, c = cgrad(:plasma, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 2", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
plot!([x_H1[i][1] for i = 1:length(x_H1)], [x_H1[i][2] for i = 1:length(x_H1)], color = :black, label = "Path 1 [-1.0,-1.0]")
plot!([x_H2[i][1] for i = 1:length(x_H2)], [x_H2[i][2] for i = 1:length(x_H2)], color = :red, label = "Path 2 [2.0, 2.0]")
plot!([x_H3[i][1] for i = 1:length(x_H3)], [x_H3[i][2] for i = 1:length(x_H3)], color = :blue, label = "Path 3 [.5,.5]")
plot!( legend=:outerbottom, legendcolumns=3)
savefig("simple2_contour_Algorithm1.png")

# Convergence Plot:
# plot(collect(1:length(f_H1)), f_H1, xlabel = "Iteration", ylabel = "Objective Function f(x)", color = :black, label = "Path 1 [-1.0-1.0]")
# plot!(collect(1:length(f_H2)), f_H2, xlabel = "Iteration", ylabel = "Objective Function f(x)", color = :red, label = "Path 2 [2.0, 2.0]")
# plot!(collect(1:length(f_H3)), f_H3, xlabel = "Iteration", ylabel = "Objective Function f(x)", color = :blue, label = "Path 3 [2.5, 2.5]")
# plot!( legend=:topright, legendcolumns=1)
# title!("Simple 2 Convergence [Quad Penalty Hook Jeeves]:")
# savefig("Simple2_Alg2_Objective Convergence.png")


# Plotting the constraint violation vs. iteration plot
# plot(1:length(v_H1), v_H1, label = "Starting Point: (0.5, 3.0)", xlabel = "Iteration", ylabel = "Constraint Violation", title = "Constraint Violation [Hook Jeeves w Quad Penalty]")
# plot!(1:length(v_H2), v_H2, label = "Starting Point: (-3.0, 2.0)")
# plot!(1:length(v_H3), v_H3, label = "Starting Point: (3.0, 2.0)")
# plot!( legend=:outerbottom, legendcolumns=3)
# savefig("simple2_Constraint Violation_Algorithm2.png")

# plot(collect(1:length(x_H1)), [max(max(0, simple2_constraints1(x_H1[i][1], x_H1[i][2])), max(0, simple2_constraints2(x_H1[i][1], x_H1[i][2]))) 
#     for i = 1:length(x_H1)], title = "Hooke Jeeves w Quad Penalty constraint violation" , xlabel = "iteration", ylabel = "c(x)",label  = "IC = [0.5,3.0]")
# plot!(collect(1:length(x_H2)), [max(max(0, simple2_constraints1(x_H2[i][1], x_H2[i][2])), max(0, simple2_constraints2(x_H2[i][1], x_H2[i][2]))) 
#     for i = 1:length(x_H2)], label = "IC = [-3.0,2.0]")
# plot!(collect(1:length(x_H3)), [max(max(0, simple2_constraints1(x_H3[i][1], x_H3[i][2])), max(0, simple2_constraints2(x_H3[i][1], x_H3[i][2]))) 
#     for i = 1:length(x_H3)], label = "IC = [3.0,2.0]")
# savefig("simple2_constraints_quad_alg2.png")