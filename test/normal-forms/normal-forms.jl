using StatsBase: median
using Plots.PlotMeasures
using LaTeXStrings
using Plots

######################################################## unit tests
function forward(name::String)

	println("Forward pass: ",name)
	hyperparameters = getParameters(targetData)

	# perform psuedoarlength continuation
	steady_states = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)

	# show and store results
	plot(steady_states,targetData)
	savefig(name)

	return true
end

function backward(name::String;n=200,idx=2)
	hyperparameters = getParameters(targetData)

	θ = range(0.03-π,π-0.03,length=n)
	L,∇L = zero(θ),zero(θ)
	target = parameters.θ[idx]

	println("Backward pass: ",name)
	for i ∈ 1:length(θ) # loss gradient across parameter grid
		parameters.θ[idx] = θ[i]

		try 
			steady_states = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters)		
			L[i],dL = ∇loss(Ref(rates),steady_states,Ref(parameters.θ),targetData.bifurcations)
			∇L[i] = dL[idx]
		catch 
			L[i] = NaN
			∇L[i] = NaN
		end
	end

	# estimate gradient using finite differences
	d̃L = vcat(NaN,diff(L)) / step(θ)

	# show and store results
	plot( θ, asinh.(d̃L),fillrange=0,label="Finite Differences",color=:darkcyan,alpha=0.5)
	plot!(θ, asinh.(∇L),            label="ForwardDiff",       color=:gold,linewidth=3)

	plot!(xlabel=L"\mathrm{parameter}, \theta",ylabel="∂Loss",right_margin=20mm)
	vline!([target], label="", color=:gold)

	plot!(twinx(),θ, asinh.(L), ylabel="Loss",color=:black,label="") |> display
	savefig(name)

	errors = abs.((∇L.-d̃L)./d̃L)
	mask = .~isnan.(errors)

	printstyled(color=:blue,"Percentage Error $(100*median(errors[mask]))% \n")
	return median(errors[mask])<0.10
end

@testset "Normal Forms" begin

	@testset "Saddle Node" begin include("saddle-node.jl")
		@test @time forward( joinpath(@__DIR__,"saddle-node.forward.pdf" ))
		@test @time backward(joinpath(@__DIR__,"saddle-node.backward.pdf"))
	end

	@testset "Pitchfork" begin include("pitchfork.jl")
		@test @time forward( joinpath(@__DIR__,"pitchfork.forward.pdf" ))
		@test @time backward(joinpath(@__DIR__,"pitchfork.backward.pdf"))
	end
end
