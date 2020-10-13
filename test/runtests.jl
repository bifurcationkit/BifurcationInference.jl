using FluxContinuation
using Flux: Momentum
using Test

using StaticArrays
using StatsBase: median
using LinearAlgebra

using Plots.PlotMeasures
using LaTeXStrings
using Plots

gr()
include("minimal/minimal.jl")


pyplot()
include("minimal/saddle-node.jl"); hyperparameters = getParameters(targetData)
heatmap( range(eps(Float64)-π,π-eps(Float64),length=100), range(eps(Float64),7,length=10),

    (α,r) -> asinh(loss(rates, SizedVector{3}(r,α,0.0), targetData, u₀, hyperparameters)),
    aspect_ratio=:equal, ylim=(eps(Float64),7), xlim=(-π,π), proj=:polar, legend=false
)

include("minimal/pitchfork.jl"); hyperparameters = getParameters(targetData)
heatmap( range(eps(Float64)-π,π-eps(Float64),length=100), range(eps(Float64),7,length=10),

    (α,r) -> asinh(loss(rates, SizedVector{2}(r,α), targetData, u₀, hyperparameters)),
    aspect_ratio=:equal, ylim=(eps(Float64),7), xlim=(-π,π), proj=:polar, legend=false
)



#
# @time trajectory = train!(rates, u₀, parameters,
#     targetData, plot_solution=5, optimiser=Momentum(0.00001), iter=50, ϵ=1.0
# )

##################### write a test to check cost in r direction



# include("scaling/scaling.jl")

# include("minimal/two-state.jl")
#
# parameters.θ .= [1.10, 0.68, 0.05, 0.73, 0.18]
# plot(rates,parameters.θ,targetData, u₀,getParameters(targetData))
#
#
# @time trajectory = train!(rates, u₀, parameters,
#     targetData, plot_solution=5, optimiser=ADAM(0.01), iter=100, ϵ=1.0
# )
# parameters.θ .= [-1.0,6.0,6.0,1.0,0.5,1.5,0.5]
# backward("test.pdf";θ=range(-1,3,length=50),idx=1)


# include("minimal/saddle-node.jl")
# hyperparameters = getParameters(targetData)
#
# pyplot()
# heatmap( range(-π,π,length=100), range(0.03,7,length=10),
#
#     (α,r) -> loss(rates, [r,α,0.0], targetData, u₀, hyperparameters),
#     aspect_ratio=:equal, xlim=(-π,π), proj=:polar, legend=false
# )
# plot!(xlabel="Cost Function")
# plot()
#
# @time trajectory = train!(rates, u₀, (θ=[5.0,-0.1,0.0], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:white, linewidth=1)
#
# @time trajectory = train!(rates, u₀, (θ=[6.0,0.5,0.0], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:white, linewidth=1)
#
# @time trajectory = train!(rates, u₀, (θ=[4.0,-2.4,0.0], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:red, linewidth=1)
# plot!(xlabel="Cost Function")
#
#
# include("minimal/pitchfork.jl")
# hyperparameters = getParameters(targetData)
# heatmap( range(-π,π,length=50), range(0.03,7,length=10),
#
#     (α,r) -> loss(rates, [r,α,0.0], targetData, u₀, hyperparameters),
#     aspect_ratio=:equal, proj=:polar, legend=false
# )
#
# @time trajectory = train!(rates, u₀, (θ=[5.0,-0.8], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:white, linewidth=1)
#
# @time trajectory = train!(rates, u₀, (θ=[6.0,2.6], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:red, linewidth=1)
#
# plot!(xlabel="Cost Function")
# savefig(joinpath(@__DIR__,"pitchfork.optimisation.pdf"))
#
# gr()
# plot(rates, [0.13832277042668428, 4.635512398276225],
#     targetData, u₀, hyperparameters, "pitchfork.optimisation.red.pdf")
