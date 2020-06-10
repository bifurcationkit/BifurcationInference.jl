using Flux,FluxContinuation,CuArrays
using StatsBase,LinearAlgebra

using Plots.PlotMeasures
using LaTeXStrings
using Test,Plots

using Flux: update!
using Parameters: @unpack
using Setfield: Lens,@lens,set

include("normal-forms/normal-forms.jl")

# function train!( parameters::NamedTuple, iter::Int, optimiser )
# 	global u₀,steady_states,hyperparameters
#
# 	for i=1:iter
#
# 		# forward pass
# 		steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)
# 		supervised = false
#
# 		if sum( branch -> sum(branch.bifurcations), steady_states) == 0
# 			hyperparameters = updateParameters(hyperparameters,steady_states)
# 		else supervised = true end
#
# 		# backward pass
# 		steady_states = cu(steady_states;nSamples=5)
# 		gradients, = gradient(parameters) do parameters
# 			L = loss(steady_states,targetData,rates,determinant,curvature,parameters,hyperparameters; supervised=supervised)
# 			println("Iteration $i\tLoss = $L\tParameters = $(parameters.θ)")
# 			return L
# 		end
#
# 		plot(steady_states,targetData) |> display
# 		update!(optimiser, parameters.θ, gradients.θ )
# 	end
# end
#
# include("two-state.jl")
# @time steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)
# plot(steady_states,targetData)
#
# train!(parameters, 10, ADAM(0.05))
# plot(steady_states,targetData)
