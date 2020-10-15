######################################################## unit tests
function forward(name::String)

	println("Forward pass: ",name)
	plot(rates,θ,targetData)

	savefig(name)
	return true
end

function backward(name::String; θi=range(0.03-π,π-0.03,length=200), idx=2)

	hyperparameters = getParameters(targetData)
	parameters = (θ=θ,p=minimum(targetData.parameter))

	L,∇L = zero(θi),zero(θi)
	target = θ[idx]

	println("Backward pass: ",name)
	for i ∈ 1:length(θi) # loss gradient across parameter grid
		parameters.θ[idx] = θi[i]

		try
			steady_states = deflationContinuation(rates,targetData.roots,parameters,hyperparameters)
			L[i],dL = ∇loss(Ref(rates),steady_states,Ref(parameters.θ),targetData.targets)
			∇L[i] = dL[idx]

		catch error
			printstyled(color=:red,"$error\n")

			L[i] = NaN
			∇L[i] = NaN
		end
	end

	# estimate gradient using finite differences
	d̃L = vcat(NaN,diff(L)) / step(θi)

	# show and store results
	plot( θi, asinh.(d̃L),fillrange=0,label="Finite Differences",color=:darkcyan,alpha=0.5)
	plot!(θi, asinh.(∇L),            label="ForwardDiff",       color=:gold,linewidth=3)

	plot!(xlabel=L"\mathrm{parameter}, \theta",ylabel="∂Loss",right_margin=20mm)
	vline!([target], label="", color=:gold)

	plot!(twinx(),θi, asinh.(L), ylabel="Loss",color=:black,label="") |> display
	savefig(name)

	errors = abs.((∇L.-d̃L)./d̃L)
	mask = .~isnan.(errors)

	printstyled(color=:blue,"Percentage Error $(100*median(errors[mask]))% \n")
	return median(errors[mask])<0.10
end

@testset "Minimal Models" begin

	@testset "Saddle Node" begin include("saddle-node.jl")
		@test @time forward( joinpath(@__DIR__,"saddle-node.forward.pdf" ))
		@test @time backward(joinpath(@__DIR__,"saddle-node.backward.pdf"))
	end

	@testset "Pitchfork" begin include("pitchfork.jl")
		@test @time forward( joinpath(@__DIR__,"pitchfork.forward.pdf" ))
		@test @time backward(joinpath(@__DIR__,"pitchfork.backward.pdf"))
	end
end
