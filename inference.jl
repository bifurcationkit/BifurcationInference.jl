using Flux, Plots, KernelDensity, LinearAlgebra, Printf
include("continuation.jl")

struct StateDensity
	parameter::AbstractRange
	density::AbstractArray
end

function infer( f::Function, θ::TrackedArray, data::StateDensity; optimiser=ADAM(0.1),
		iter::Int=100, u₀::TrackedArray=param([-2.0,0.0]), maxSteps::Int=1000 )

	# setting initial hyperparameters
	p₀ = param(minimum(data.parameter))
	pMax,ds = maximum(data.parameter), step(data.parameter)
	u₀,P,U = continuation( f,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds, maxSteps=maxSteps )

	function predictor()

		u₀,P,U = continuation( f,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds, maxSteps=maxSteps )
		kernel = kde(P,data.parameter,bandwidth=1.4*ds)
		return kernel.density
	end

	loss() = norm( predictor() .- data.density )

	function progress()
		@printf("Loss = %f, θ = %f,%f,%f\n", loss(), θ.data...)
		plot( P.data,U.data,
			label="inferred", color="darkblue",linewidth=3)
		plot!( data.parameter,predictor().data,
			label="inferred", color="darkblue")
		plot!( data.parameter,data.density,
			label="target", color="gold",
			xlabel="parameter, p", ylabel="steady state") |> display
	end

	@time Flux.train!(loss, Flux.Params([θ]),
		Iterators.repeated((),iter), optimiser;
		cb=progress)
end
