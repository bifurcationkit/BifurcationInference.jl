include("../src/patches/KernelDensity.jl")
using FluxContinuation: continuation,unpack
using PseudoArcLengthContinuation: plotBranch

using Flux, KernelDensity, Plots, LinearAlgebra, Printf
using Flux.Tracker: update!

struct StateDensity
    parameter::AbstractRange
    density::AbstractArray
end

function infer( f::Function, J::Function, u₀::Vector{TrackedReal{T}}, θ::TrackedArray, data::StateDensity;
        optimiser=ADAM(0.05), iter::Int=100, maxSteps::Int=1000, maxIter::Int=1000 ) where T

    # setting initial hyperparameters
    global p₀,pMax,ds
    p₀ = param(minimum(data.parameter))
    pMax,ds = maximum(data.parameter),step(data.parameter)

    function predictor()
        global bifurcations,prediction,f,J,u₀,p₀

        bifurcations,u₀ = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=10*ds,
            maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=false )
        prediction = kde( unpack(bifurcations)[1], data.parameter, bandwidth=ds)

        return bifurcations,prediction
    end

    function loss()

        bifurcations,prediction = predictor()
        ℒ = norm( prediction.density .- data.density )

        @printf("points %i    ",size(bifurcations.branch.u)...)
        @printf("Loss = %f\n", ℒ)

        return ℒ
    end

    @time Flux.train!(loss, Flux.Params([θ]),
        Iterators.repeated((),iter), optimiser)
end

function predictor()
    global bifurcations,prediction,f,J,u₀,p₀

    bifurcations,u₀ = continuation( f,J,u₀,p₀; pMin=p₀-ds, pMax=pMax, ds=10*ds,
        maxSteps=maxSteps, maxIter=maxIter, computeEigenValues=false )
    density = kde( unpack(bifurcations)[1], data.parameter, bandwidth=ds)

    return bifurcations,density
end

function loss()

    bifurcations,prediction = predictor()
    ℒ = norm( prediction.density .- data.density )

    @printf("points %i    ",size(bifurcations.branch.u)...)
    @printf("Loss = %f\n", ℒ)

    return ℒ
end

function progress()

    plotBranch( bifurcations, label="inferred")
    plot!( data.parameter, prediction.density, label="inferred", color="darkblue")
    plot!( data.parameter, data.density, label="target", color="gold",
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
