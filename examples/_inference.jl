include("../src/patches/KernelDensity.jl")
using FluxContinuation: continuation,unpack
using PseudoArcLengthContinuation: plotBranch

using Flux, KernelDensity, Plots, LinearAlgebra, Printf
using Flux.Tracker: update!,TrackedReal

struct StateDensity
    parameter::AbstractRange
    stable::AbstractArray
	unstable::AbstractArray
end

function infer( f::Function, J::Function, u₀::Vector{TrackedReal{T}}, θ::TrackedArray, data::StateDensity;
        optimiser=ADAM(0.05), iter::Int=100, maxSteps::Int=1000, maxIter::Int=1000, tol=1e-10 ) where T

    # setting initial hyperparameters
    global p₀,pMax,ds
    p₀ = param(minimum(data.parameter))
    pMax,ds = maximum(data.parameter),step(data.parameter)

    function predictor()
        global bifurcations,stable,unstable,f,J,u₀,p₀

        bifurcations,u₀ = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds,
            maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=true, tol=tol )

        stable = kde( bifurcations.branch[1,bifurcations.stability], data.parameter, bandwidth=ds)
		unstable = kde( bifurcations.branch[1,.!bifurcations.stability], data.parameter, bandwidth=ds)

        return bifurcations,stable,unstable
    end

    function loss()
		global bifurcations,stable,unstable

        bifurcations,stable,unstable = predictor()
        ℒ = norm( stable.density .- data.stable ) + norm( unstable.density .- data.unstable )

		# if length(bifurcations.bifpoint) > 0
		# 	error = norm(bifurcations.bifpoint[1].param-0.5)+norm(bifurcations.bifpoint[2].param+0.5)
		# 	λ = 1/(1+error)
		# else
		# 	λ = 0.0
		# end

        @printf("points %i    ",size(bifurcations.branch.u)...)
        @printf("Loss = %f\n", ℒ)

        return ℒ#*(1-λ)
    end

    @time Flux.train!(loss, Flux.Params([θ]),
        Iterators.repeated((),iter), optimiser, cb=progress)
end

function predictor()
	global bifurcations,stable,unstable,f,J,u₀,p₀

	bifurcations,u₀ = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=ds,
		maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=true, tol=tol )

	stable = kde( bifurcations.branch[1,bifurcations.stability], data.parameter, bandwidth=ds)
	unstable = kde( bifurcations.branch[1,.!bifurcations.stability], data.parameter, bandwidth=ds)

	return bifurcations,stable,unstable
end

function loss()
	global bifurcations,stable,unstable

	bifurcations,stable,unstable = predictor()
	ℒ = norm( stable.density .- data.stable ) + norm( unstable.density .- data.unstable )

	# if length(bifurcations.bifpoint) > 0
	# 	error = norm(bifurcations.bifpoint[1].param-0.5)+norm(bifurcations.bifpoint[2].param+0.5)
	# 	λ = 1/(1+error)
	# else
	# 	λ = 0.0
	# end

    # @printf("points %i    ",size(bifurcations.branch.u)...)
    # @printf("Loss = %f\n", ℒ)

    return ℒ#*(1-λ)
end

function progress()

    plotBranch( bifurcations, label="" )
    plot!( data.parameter, stable.density, label="inferred", color="darkblue")
	plot!( data.parameter, unstable.density, label="", color="darkblue", linestyle=:dash)
	plot!( data.parameter, data.unstable, label="", color="orange", linestyle=:dash)
    plot!( data.parameter, data.stable, label="target", color="orange",
        xlabel="parameter, p", ylabel="steady state") |> display
end

function lossAt(params...)
	copyto!(θ.data,[params...])
	try
		return Tracker.data(loss())
	catch
		return NaN
	end
end
