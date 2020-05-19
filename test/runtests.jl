using Flux,FluxContinuation
using StatsBase,LinearAlgebra
using StatsBase: median

using Plots.PlotMeasures
using Test,Plots

using Flux: Params,update!
using Parameters: @unpack
using Setfield: Lens,@lens,set

######################################################## unit tests
function test_predictor() global u₀,steady_states,hyperparameters
	steady_states,u₀ = deflationContinuation(rates,rates_jacobian,u₀,θ,(@lens _.p),hyperparameters)
	hyperparameters = updateParameters(hyperparameters,steady_states)

	plot(steady_states,targetData)
	return true
end

function evaluate_gradient(x; idx=2) global θ,u₀,steady_states,hyperparameters
	θ,L = set(θ, (@lens _.z[idx]), x), NaN

	try
		dθ = gradient(Params(θ)) do

			steady_states,u₀ = deflationContinuation(rates,rates_jacobian,u₀,θ,(@lens _.p),hyperparameters)
			hyperparameters = updateParameters(hyperparameters,steady_states)

			L = loss(steady_states,targetData,curvature)
			return L
		end

		return [ L, dθ[GlobalRef(Main,:θ)].z[idx] ]
	catch
		return [NaN,NaN]
	end
end

function test_gradients( ; tol=0.1,ϕ=range(0.03,2π-0.03,length=100))

	L,dL = zeros(length(ϕ)), zeros(length(ϕ))
	for i = 1:length(ϕ)
		L[i],dL[i] = evaluate_gradient(ϕ[i])
	end
	finite_differences = vcat(diff(L)/step(ϕ),NaN)

	plot([],[],label="",linewidth=2,ylim=(-20,10),xlabel="Parameter",right_margin=15mm,size=(500,300),legend=:bottomright)
	plot!(ϕ,sign.(finite_differences).*log.(abs.(finite_differences)),label="Finite Differences",color="gold",fillrange=0,alpha=0.5)
	plot!(ϕ,sign.(dL).*log.(abs.(dL)),label="Zygote AutoDiff",color="lightblue",fillrange=0,alpha=0.6) |> display

	errors = abs.(dL-finite_differences)
	mask = .~isnan.(errors)
	return median(errors[mask]) < tol
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

# include("two-state.jl")
# θ.z[1] = 6.
# test_predictor()
#
# infer( rates, rates_jacobian, curvature, u₀, (@lens _.p),
# 	targetData; iter=50, optimiser=ADAM(0.01))

@testset "Normal Forms" begin

    include("saddle-node.jl")
    @test test_predictor()
    @test test_gradients()

    include("pitchfork.jl")
    @test test_predictor()
    @test test_gradients()
end

@testset "Inference" begin

	# include("two-state.jl")
    # @test test_predictor()
    #@test test_gradients()
	#
	# include("cell-cycle.jl")
    # @test test_predictor()
    # @test test_gradients()
end
