using Flux, Plots, KernelDensity, LinearAlgebra, Printf
include("continuation.jl")

struct StateDensity
	parameter::AbstractRange
	density::AbstractArray
end

function infer( f::Function, θ::TrackedArray, data::StateDensity; optimiser=ADAM(0.1),
		iter::Int=100, u₀::AbstractFloat=-2.0, uRange::AbstractFloat=1e3 )

	# setting initial hyperparameters
	p₀,pMax,ds = minimum(data.parameter), maximum(data.parameter), step(data.parameter)
	u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)
	U,P = continuation( (u,p)->f(u,p).data ,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )

	function predictor()

		# predict parameter curve
		U,P = continuation( f,u₀,p₀; ds=ds, pMax=pMax, uRange=uRange )
		u₀,_,_= tangent( (u,p)->f(u,p).data ,u₀,p₀; ds=ds)

		# state density as multi-stability label
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
