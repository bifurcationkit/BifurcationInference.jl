using Flux,FluxContinuation,CuArrays
using StatsBase,LinearAlgebra
using StatsBase: median

using Plots.PlotMeasures
using Test,Plots

using Flux: Params,update!
using Zygote: forward_jacobian

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

	# backward pass
	gradients, = gradient(params) do params
		loss(steady_states,targetData,rates,determinant,curvature,params)
	end

	# central differences
	L₊ = loss(steady_states,targetData,rates,determinant,curvature,set(params,(@lens _.θ[idx]), params.θ[idx]+T(dx)/2))
	L₋ = loss(steady_states,targetData,rates,determinant,curvature,set(params,(@lens _.θ[idx]), params.θ[idx]-T(dx)/2))

	return (L₊+L₋)/2, (L₊-L₋)/dx, isnothing(gradients) ? NaN : gradients.θ[idx]
end

function test_gradients(name::String;n=200)
	x = range(0.03,2π-0.03,length=n)
	L,d̃L,dL = zero(x), zero(x), zero(x)

	for i in 1:length(x)
		L[i],d̃L[i],dL[i] = test_gradient(( θ=[5.0,x[i],0.0], p=-2.0))
	end

	plot( x,d̃L,fillrange=0,label="Central Differences",color=:darkcyan,alpha=0.5)
	plot!(x,dL,fillrange=0,label="Zygote",color=:gold,alpha=0.5)
	plot!(xlabel="parameter",ylabel="loss gradient",ylim=(-30,30))

	plot!(twinx(),x,L,ylabel="loss",color=:black,label="")|> display
	savefig(name)

	errors = abs.((dL.-d̃L)/d̃L)
	printstyled(color=:orange,"Zygote Percentage Error $(100*mean(errors[.~isnan.(errors)]))% \n")
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

function infer( rates::Function, rates_jacobian::Function, curvature::Function,
		u₀::Vector{Array{T,2}}, paramlens::Lens, targetData::StateDensity;
        optimiser=Momentum(), callback::Function=()->(), iter::Int=10, maxIter::Int=10, tol::Float64=1e-12 ) where T

    hyperparameters = getParameters(targetData; maxIter=maxIter, tol=tol)
	function loss( ; offset::Number=5.0) global u₀,steady_states,hyperparameters

		steady_states,u₀ = deflationContinuation(rates,rates_jacobian,u₀,θ,paramlens,hyperparameters)
		hyperparameters = updateParameters(hyperparameters,steady_states)

		predictions = map( branch -> map(
			bifurcation -> bifurcation.parameter, branch.bifurcations ),
				steady_states)

	    predictions = vcat(predictions...)
		if length(predictions) > 0  # supervised signal

	    	error = norm(minimum(abs.(targetData.bifurcations.-predictions'),dims=2))
			return tanh(log(error))-offset

		else # unsupervised signal
			total_curvature = sum( branch-> sum(
		        	abs.(curvature.(branch.state,branch.parameter)).*abs.(branch.ds)),
		        steady_states )

			return -log(1+total_curvature)
		end
	end

    @time train!(loss, iter, optimiser; callback=callback)
end

function train!(loss::Function, iter::Int, optimiser; callback::Function=()->() )
	global θ # @gszep(todo) temporary work-around for NamedTuple grads

	for i=1:iter

		gradients = gradient(Params(θ)) do
			L = loss()
			println("Iteration = $i, Loss(z) = $L, z = $(θ.z)")
			return L
		end

		# update global parameters
		plot(steady_states,targetData)
		update!(optimiser, θ.z, gradients[GlobalRef(Main,:θ)].z)
	end
end
