using Flux,FluxContinuation
using StatsBase: median
using Test,Plots

using Parameters: @unpack
using Setfield: @lens,set

######################################################## unit tests
function test_predictor() global u₀,steady_states,hyperparameters
	steady_states,u₀ = deflationContinuation(rates,rates_jacobian,u₀,θ,(@lens _.p),hyperparameters)
	hyperparameters = updateParameters(hyperparameters,steady_states)

	plot(steady_states,data)
	return true
end

function evaluate_gradient(x; index=1) global θ,u₀,steady_states,hyperparameters
	θ = set(θ, (@lens _.α), x)

	try
		dθ = gradient(params(θ)) do
			steady_states,u₀ = deflationContinuation(rates,rates_jacobian,u₀,θ,(@lens _.p),hyperparameters)
			hyperparameters = updateParameters(hyperparameters,steady_states)
			loss(steady_states,data,curvature)
		end

		return [ loss(steady_states,data,curvature), dθ[GlobalRef(Main,:θ)][index] ]
	catch
		return [NaN,NaN]
	end
end

using Plots.PlotMeasures
function test_gradients(tol=0.01) global gradients

    gradients = hcat(evaluate_gradient.(ϕ)...)
    L,dL = gradients[1,:], gradients[2,:]
    finite_differences = vcat(diff(L)/step(ϕ),NaN)

    step_truncate = 50
    mask = abs.(finite_differences).>step_truncate
    finite_differences[mask] .= NaN

    plot(fill(θ.α₀,2),[-1,1.4],label="",color="gold",linewidth=2,ylim=(-5,5),
    	xlabel="Parameter",ylabel="Objective",
    	right_margin=15mm,size=(500,500))

    right_axis = twinx(); plot!(right_axis,label="",
    	ylim=(-35,20),ylabel="Gradient",
    	legend=:bottomleft)

    plot!(ϕ,L,label="",color="black")
    plot!(right_axis,ϕ,finite_differences,label="Finite Differences",color="gold",fillrange=0,alpha=0.5)
    plot!(right_axis,ϕ,dL,label="Zygote AutoDiff",color="lightblue",fillrange=0,alpha=0.6) |> display

	errors = abs.(dL-finite_differences)
	mask = .~isnan.(errors)
	return median(errors[mask]) < tol
end

@testset "Normal Forms" begin

    include("saddle-node.jl")
    @test test_predictor()
    @test test_gradients()

    # include("pitchfork.jl")
    # @test test_predictor()
    # @test test_gradients()
	#
	# include("two-state.jl")
    # @test test_predictor()
    # @test test_gradients()
	#
	# include("cell-cycle.jl")
    # @test test_predictor()
    # @test test_gradients()

end
