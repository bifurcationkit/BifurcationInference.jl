using Flux, Plots, KernelDensity, LinearAlgebra, Printf
include("continuation.jl")

struct StateDensity
	parameter::AbstractRange
	density::AbstractArray
end

function infer( f::Function, J::Function, θ::TrackedArray, data::StateDensity; optimiser=ADAM(0.1),
		iter::Int=100, u₀=[-2.0,0.0], maxSteps::Int=1000, maxIter::Int=1000 )

	# setting initial hyperparameters
	P,U = param([0.0]),param([0.0])
	p₀ = param(minimum(data.parameter))
	pMax,ds = maximum(data.parameter), step(data.parameter)

	function predictor()

		u₀,P,U = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds, maxSteps=maxSteps, maxIter=maxIter )
		kernel = kde(P,data.parameter,bandwidth=1.4*ds)
		return kernel.density
	end

	loss() = norm( predictor() .- data.density )

	function progress()
		@printf("Loss = %f, θ = [%f,%f,%f], u₀ = [%f,%f]\n", loss(), θ.data...,u₀...)
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
