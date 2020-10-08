using FluxContinuation
using Test

include("normal-forms/normal-forms.jl")


# include("normal-forms/saddle-node.jl")
# pyplot()

# hyperparameters = getParameters(targetData)
# heatmap( range(-π,π,length=50), range(0.03,7,length=10),

#     (α,r) -> loss(rates, [r,α,0.0], targetData, u₀, hyperparameters),
#     aspect_ratio=:equal, proj=:polar, legend=false
# )

# @time trajectory = train!(rates, u₀, (θ=[5.0,-0.1,0.0], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:white, linewidth=1)

# @time trajectory = train!(rates, u₀, (θ=[6.0,0.5,0.0], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:white, linewidth=1)

# @time trajectory = train!(rates, u₀, (θ=[4.0,-2.4,0.0], p=minimum(targetData.parameter)), targetData )
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:black, linewidth=3)
# plot!( map( x -> x[2], trajectory), map( x -> x[1], trajectory), color=:white, linewidth=1)
# plot!(xlabel="Cost Function")

# include("normal-forms/pitchfork.jl")
# parameters = (θ=[5.0,4.36], p=-2.0)
# @time train(parameters, 100, Momentum(0.001))

# include("normal-forms/saddle-node.jl")
# parameters = (θ=[5.0,-0.1,0.0], p=-2.0)
# train!(parameters, 80, Momentum(0.001))
#
# include("normal-forms/saddle-node.jl")
# parameters = (θ=[5.0, 5.35,0.6], p=-2.0)
# train!(parameters, 120, Momentum(0.001))

# include("normal-forms/pitchfork.jl")
# parameters = (θ=[5.0,-0.1], p=-2.0)
# train!(parameters, 200, Momentum(0.001))
#
# include("normal-forms/saddle-node.jl")
# parameters = (θ=randn(3), p=-2.0)
# train!(parameters, 4000, Momentum(0.001))