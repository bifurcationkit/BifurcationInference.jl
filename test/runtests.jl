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
	global u₀,steady_states,hyperparameters

	for i=1:iter

		# forward pass
		steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)
		hyperparameters = updateParameters(hyperparameters,steady_states)
		@unpack state,parameter,ds = cu(steady_states)

		# backward pass
		gradients, = gradient(parameters) do parameters
			L = loss(steady_states,targetData,rates,determinant,curvature,parameters,hyperparameters)
			println("Iteration $i\tLoss = $L\tParameters = $(parameters.θ)")
			return L
		end
		plot(steady_states,targetData) |> display
		update!(optimiser, parameters.θ, gradients.θ )
	end
end

# include("two-state.jl")
# @time test_predictor(parameters,"test/two-state.forward.pdf")
# @time test_gradient(parameters)
#
# train!(parameters, 1, ADAM(0.05))
# plot(steady_states,targetData)
