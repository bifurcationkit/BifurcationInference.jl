include("../src/patches/KernelDensity.jl")
using FluxContinuation: continuation
using Flux, Zygote, KernelDensity

using PseudoArcLengthContinuation: plotBranch,ContinuationPar,NewtonPar,DefaultLS,DefaultEig
using Plots, LinearAlgebra, Printf

struct StateDensity
    parameter::AbstractRange
    density::AbstractArray
end

function infer( f::Function, J::Function, u₀::Vector{T}, θ::AbstractArray, data::StateDensity;
        optimiser=ADAM(0.05), iter::Int=100, maxSteps::Int=1000, maxIter::Int=1000, tol=1e-10 ) where T

    global parameters
    parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=maxSteps,

			newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
			verbose=false,maxIter=maxIter,tol=tol),

		computeEigenValues = false)

    function predictor()
        global bifurcations,density,f,J,u₀

        bifurcations,u₀ = continuation( f,J,u₀, parameters )
        prediction = kde( bifurcations.branch[1,:], data.parameter, bandwidth=parameters.ds)

        return bifurcations,prediction
    end

    function loss()
		global bifurcations,prediction
        bifurcations,prediction = predictor()
        return norm( prediction.density .- data.density )
    end

    @time Flux.train!(loss, Zygote.Params([θ]), iter, optimiser, cb=progress)
end

function predictor()
	global bifurcations,density,f,J,u₀

	bifurcations,u₀ = continuation( f,J,u₀, parameters )
    prediction = kde( bifurcations.branch[1,:], data.parameter, bandwidth=parameters.ds)

	return bifurcations,prediction
end

function loss()
	global bifurcations,prediction
    bifurcations,prediction = predictor()
    return norm( prediction.density .- data.density )
end

function progress()
    plotBranch( bifurcations, label="inferred")
    plot!( data.parameter, data.density,
        label="target", color="gold",
        xlabel="parameter, p", ylabel="steady state") |> display
end

function lossAt(params...)
	copyto!(θ,[params...])
	try
		return loss()
	catch
		return NaN
	end
end
