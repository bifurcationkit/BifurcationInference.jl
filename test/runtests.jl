using Flux,FluxContinuation,CuArrays
using StatsBase,LinearAlgebra

using Plots.PlotMeasures
using Test,Plots

using Flux: update!
using Parameters: @unpack
using Setfield: Lens,@lens,set

######################################################## unit tests
function test_predictor(params::NamedTuple, name::String) where T<:Number
	global u₀,steady_states,hyperparameters

	steady_states,u₀ = deflationContinuation(rates,u₀,params,(@lens _.p),hyperparameters; verbosity=1)
	hyperparameters = updateParameters(hyperparameters,steady_states)

	plot(steady_states,targetData)
	savefig(name)
	return true
end

function test_gradient(params::NamedTuple; idx=2, dx::T = 1e-6) where T<:Number
	global u₀,steady_states,hyperparameters

	# forward pass
	steady_states,u₀ = deflationContinuation(rates,u₀,params,(@lens _.p),hyperparameters)
	hyperparameters = updateParameters(hyperparameters,steady_states)
	@unpack state,parameter = cu(steady_states)

	# backward pass
	gradients, = gradient(params) do params
		loss(steady_states,state,parameter,targetData,rates,determinant,curvature,params)
	end

	# central differences
	L₊ = loss(steady_states,state,parameter,targetData,rates,determinant,curvature,set(params,(@lens _.θ[idx]), params.θ[idx]+T(dx)/2))
	L₋ = loss(steady_states,state,parameter,targetData,rates,determinant,curvature,set(params,(@lens _.θ[idx]), params.θ[idx]-T(dx)/2))

	return (L₊+L₋)/2, (L₊-L₋)/dx, isnothing(gradients) ? NaN : gradients.θ[idx]
end

function test_gradients(name::String;n=200)
	x = range(0.03,2π-0.03,length=n)
	L,d̃L,dL = zero(x), zero(x), zero(x)

	for i in 1:length(x)
		L[i],d̃L[i],dL[i] = test_gradient(( θ=[5.0,x[i],0.0], p=-2.0))
	end

	plot( x, d̃L,fillrange=0,label="Central Differences",color=:darkcyan,alpha=0.5)
	plot!(x, dL,label="Zygote",color=:gold,linewidth=3)
	plot!(xlabel="parameter",ylabel="∂Loss",right_margin=20mm,ylim=(-30,30))

	plot!(twinx(),x, sign.(L).*log.(abs.(L)), ylabel="SymLog Loss",color=:black,label="")|> display
	savefig(name)

	errors = abs.((dL.-d̃L)/d̃L)
	printstyled(color=:blue,"Zygote Percentage Error $(100*mean(errors[.~isnan.(errors)]))% \n")
	return all(errors[.~isnan.(errors)].<0.05)
end

@testset "Normal Forms" begin

	@testset "Saddle Node" begin include("saddle-node.jl")
		@test @time test_predictor(parameters,"test/saddle-node.predictor.pdf")
		@test @time test_gradients("test/saddle-node.gradients.pdf")
	end

	@testset "Pitchfork" begin include("pitchfork.jl")
		@test @time test_predictor(parameters,"test/pitchfork.predictor.pdf")
	    @test @time test_gradients("test/pitchfork.gradients.pdf")
	end
end

function train!( parameters::NamedTuple, iter::Int, optimiser )
	global u₀,steady_states,hyperparameters

	for i=1:iter

		# forward pass
		steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)
		hyperparameters = updateParameters(hyperparameters,steady_states)

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

include("two-state.jl")
@time test_predictor(parameters,"test/two-state.predictor.pdf")
@time test_gradient(parameters)

#parameters = ( θ=[4.0,π/4,1.0],p=minimum(targetData.parameter))

train!(parameters, 1, ADAM(0.05))
plot(steady_states,targetData)



test_predictor(( θ=[4.0,3π/2-0.01,0.0], p=-5),"test/pitchfork.predictor.pdf")


include("saddle-node.jl")
parameters = ( θ=[4.0,π/4,1.0],p=minimum(targetData.parameter))

train!(parameters, 200, ADAM(0.05))
plot(steady_states,targetData)

include("pitchfork.jl")
parameters = ( θ=[5.0,π+1/2,1.0],p=minimum(targetData.parameter))

train!(parameters, 100, ClipNorm(0.01))
plot(steady_states,targetData)
