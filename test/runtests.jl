using Flux, FluxContinuation
using StatsBase: median
using Test,Plots

######################################################## unit tests
function test_predictor() global u₀,steady_states,parameters
	steady_states,u₀ = deflationContinuation(f,J,u₀,parameters)
	parameters = updateParameters(parameters,steady_states)

	plot(steady_states,data)
	return true
end

function evaluate_gradient(x...; index=1) global θ,u₀,steady_states,parameters
	copyto!(θ,[x...])

	dθ = gradient(params(θ)) do
		steady_states,u₀ = deflationContinuation(f,J,u₀,parameters )
		parameters = updateParameters(parameters,steady_states)
		loss(steady_states,data,K)
	end

	return [ loss(steady_states,data,K), dθ[θ][index] ]
end

using Plots.PlotMeasures
function test_gradients(tol=0.01)

    gradients = hcat(evaluate_gradient.(ϕ,r)...)
    L,dL = gradients[1,:], gradients[2,:]
    finite_differences = vcat(diff(L)/step(ϕ),NaN)

    step_truncate = 50
    mask = abs.(finite_differences).>step_truncate
    finite_differences[mask] .= NaN

    plot(fill(optimum,2),[-1,1.4],label="",color="gold",linewidth=2,ylim=(-5,5),
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

@testset "normal forms" begin

    include("saddle-node.jl")
    @test test_predictor()
    @test test_gradients()

    #include("pitchfork.jl")
    @test test_predictor()
    #@test test_gradients()

end
