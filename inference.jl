using Flux, Plots, LinearAlgebra, Printf
include("patches/PseudoArcLengthContinuation.jl")
include("patches/KernelDensity.jl")

struct StateDensity
	parameter::AbstractRange
	density::AbstractArray
end

function infer( f::Function, J::Function, θ::TrackedArray, data::StateDensity; optimiser=ADAM(0.1),
		iter::Int=100, u₀=param.([-2.0,0.0]), maxSteps::Int=1000, maxIter::Int=1000 )

	# setting initial hyperparameters
	p₀ = param(minimum(data.parameter))
	pMax,ds = maximum(data.parameter), step(data.parameter)

	bifurcations, = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds,
		maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=true )

	prediction = kde( unpack(bifurcations)[1], data.parameter, bandwidth=1.4*ds)
	u₀ = initial_state(bifurcations)

	function predictor()

		bifurcations, = continuation( f,J,u₀, p₀; pMin=p₀-ds, pMax=pMax, ds=ds,
			maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=true )

		density = kde( unpack(bifurcations)[1], data.parameter, bandwidth=1.4*ds)
		u₀ = initial_state(bifurcations)

		return bifurcations,density
	end

	function loss()
		_,prediction = predictor()
		ℒ = norm( prediction.density .- data.density )
		return ℒ
	end

	function progress()
		@printf("Loss = %f, θ = [%f,%f,%f], u₀ = [%f,%f]\n", loss(), θ.data...,u₀...)
		plotBranch( bifurcations, label="inferred")
		plot!( data.parameter, prediction.density, label="inferred", color="darkblue")
		plot!( data.parameter,data.density, label="target", color="gold",
			xlabel="parameter, p", ylabel="steady state") |> display
	end

	@time Flux.train!(loss, Flux.Params([θ]),
		Iterators.repeated((),iter), optimiser;
		cb=progress)
end
