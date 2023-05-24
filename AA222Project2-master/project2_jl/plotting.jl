using Plots
using LinearAlgebra

include("helpers.jl")
include("simple.jl")
# Simple 1:
function simple1(x1, x2)
    return -x1 * x2 + 2.0 / (3.0 * sqrt(3.0))
end
function simple1_gradient(x::Vector)
    return [-x[2], -x[1]]
end
# function simple1_constraints(x::Vector)
#     return [x[1] + x[2]^2 - 1,
#             -x[1] - x[2]]
# end
function simple1_constraints1(x1,x2)
    return x1 + x2^2 - 1
end
function simple1_constraints2(x1,x2)
    return -x1 - x2
end
function simple1_init()
    return rand(2) * 2.0
end

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
#ρ = 10
#mod_f1(x) = simple1(x) + ρ*sum(x -> x > 0 ? x : 0, simple1_constraints(x))

# Quad Penalty
ρ2 = 9999999999^30
mod_f1(x) = simple1(x) + ρ2*sum(max.(0,simple2_constraints(x)).^2)

# Plotting for Simple 1:
x_H1, f_H1 =hooke_jeeves2(mod_f1, [-1.0, -1.0], simple1_gradient, simple1_constraints, 2000, 0.1, 0.5 )
# x_H1_Alg2, f_H1_Alg2 =hooke_jeeves2(mod_f_Alg2, [-1.5, -1.5], simple1_gradient, simple1_constraints, 2000, 0.1, 0.5 )
x_H2, f_H2 =hooke_jeeves2(mod_f1, [2.0, 2.0], simple1_gradient, simple1_constraints, 6000, 0.1, 0.5 )
x_H3, f_H3 =hooke_jeeves2(mod_f1, [0.5, 0.5], simple1_gradient, simple1_constraints, 10000, 0.1, 0.5 )
xr = -3:0.1:3
yr = -3:0.1:3
contourf(xr, yr, simple1, levels = LinRange(-5,10,21), colorbar = false, c = cgrad(:viridis, rev = true), legend = false, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 1", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
contour!(xr, yr, simple1_constraints1, levels = [0], colorbar = false, c = cgrad(:plasma, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 1", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
contour!(xr, yr, simple1_constraints2, levels = [0], colorbar = false, c = cgrad(:plasma, rev = true), legend = true, 
        xlims = (-3,3), ylims = (-3,3), title = "Contour Plot for Simple 1", xlabel = "x1 ", ylabel = "x2", aspectratio = :equal, clim = (-10,10))
plot!([x_H1[i][1] for i = 1:length(x_H1)], [x_H1[i][2] for i = 1:length(x_H1)], color = :black, label = "Path 1 [-1.0,-1.0]")
plot!([x_H2[i][1] for i = 1:length(x_H2)], [x_H2[i][2] for i = 1:length(x_H2)], color = :red, label = "Path 2 [2.0, 2.0]")
plot!([x_H3[i][1] for i = 1:length(x_H3)], [x_H3[i][2] for i = 1:length(x_H3)], color = :blue, label = "Path 3 [.5,.5]")
plot!( legend=:outerbottom, legendcolumns=3)
savefig("simple1_contour_Algorithm2.png")