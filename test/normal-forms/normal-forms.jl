######################################################## unit tests
function forward(name::String) where T<:Number
	global u₀,steady_states,hyperparameters

	steady_states,u₀ = deflationContinuation(rates,u₀,parameters,(@lens _.p),hyperparameters; verbosity=1)
	hyperparameters = updateParameters(hyperparameters,steady_states)

	plot(steady_states,targetData)
	savefig(name)
	return true
end

function backward(name::String;n=200)
	x = range(0.03-π,π-0.03,length=n)
	L,d̃L,dL = zero(x), zero(x), zero(x)

	for i in 1:length(x)
		try
			L[i],d̃L[i],dL[i] = gradients(( θ=[5.0,x[i],0.0], p=-2.0))
		catch
			L[i],d̃L[i],dL[i] = NaN,NaN,NaN
		end
	end

	plot( x, asinh.(d̃L),fillrange=0,label="Central Differences",color=:darkcyan,alpha=0.5)
	plot!(x, asinh.(dL),label="Zygote",color=:gold,linewidth=3)
	plot!(xlabel=L"\mathrm{inference\,\,\,parameter}, \theta",ylabel="∂Loss",right_margin=20mm)

	plot!(twinx(),x, asinh.(L), ylabel="Loss",color=:black,label="")|> display
	savefig(name)

	errors = abs.((dL.-d̃L)/d̃L)
	printstyled(color=:blue,"Zygote Percentage Error $(100*mean(errors[.~isnan.(errors)]))% \n")
	return all(errors[.~isnan.(errors)].<0.05)
end

function gradients(params::NamedTuple; idx=2, dx::T = 1e-6) where T<:Number
	global u₀,steady_states,hyperparameters

	# forward pass
	steady_states,u₀ = deflationContinuation(rates,u₀,params,(@lens _.p),hyperparameters)
	hyperparameters = updateParameters(hyperparameters,steady_states)
	supervised = sum( branch -> sum(branch.bifurcations), steady_states) == 0 ? false : true

	# backward pass
	steady_states = cu(steady_states)
	gradients, = gradient(params) do params
		loss(steady_states,likelihood,curvature,rates,params,hyperparameters,supervised)
	end

	# central differences
	L₊ = loss(steady_states,likelihood,curvature,rates,set(params,(@lens _.θ[idx]), params.θ[idx]+T(dx)/2),hyperparameters,supervised)
	L₋ = loss(steady_states,likelihood,curvature,rates,set(params,(@lens _.θ[idx]), params.θ[idx]-T(dx)/2),hyperparameters,supervised)

	return (L₊+L₋)/2, (L₊-L₋)/dx, isnothing(gradients) ? NaN : gradients.θ[idx]
end

@testset "Normal Forms" begin

	@testset "Saddle Node" begin include("saddle-node.jl")
		@test @time forward("normal-forms/saddle-node.forward.pdf")
		@test @time backward("normal-forms/saddle-node.backward.pdf")
	end

	@testset "Pitchfork" begin include("pitchfork.jl")
		@test @time forward("normal-forms/pitchfork.forward.pdf")
	    @test @time backward("normal-forms/pitchfork.backward.pdf")
	end
end
