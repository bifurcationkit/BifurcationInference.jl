using Flux,FluxContinuation,CuArrays
using StatsBase,LinearAlgebra

using Plots.PlotMeasures
using LaTeXStrings
using Test,Plots

using Flux: update!
using Parameters: @unpack
using Setfield: Lens,@lens,set

include("normal-forms/normal-forms.jl")
function train!( parameters::NamedTuple, iter::Int, optimiser )
	global u₀,steady_states,hyperparameters,supervised

	for i=1:iter
		try
			steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)
			hyperparameters = updateParameters(hyperparameters,steady_states)
			supervised = sum( branch -> sum(branch.bifurcations),steady_states) == 0 ? false : true
		catch
			printstyled(color=:red,"Iteration $i\tSkipped\n")
		end

		Loss = NaN
		steady_states = cu(steady_states)

		gradients, = gradient(parameters) do parameters
			Loss = loss(steady_states,likelihood,curvature,rates,parameters,hyperparameters,supervised)
		end

		printstyled(color=:yellow,"Iteration $i\tLoss = $Loss\n")
		println("Parameters\t$(parameters.θ)")
		println("Gradients\t$(gradients.θ)")
		update!(optimiser, parameters.θ, gradients.θ )
	end

	# plot final model
	steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)
	plot(steady_states,targetData)
end

include("normal-forms/saddle-node.jl")
parameters = (θ=[5.0,3.2,0.0], p=-2.0)
@time train!(parameters, 100, Momentum(0.001))

include("normal-forms/pitchfork.jl")
parameters = (θ=[5.0,4.36], p=-2.0)
@time train!(parameters, 100, Momentum(0.001))

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
