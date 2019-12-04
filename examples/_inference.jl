include("../src/patches/KernelDensity.jl")
using PseudoArcLengthContinuation: plotBranch,ContinuationPar,NewtonPar,DefaultLS,DefaultEig
using FluxContinuation: continuation
using Flux, Zygote, Plots, Printf

using KernelDensity: kde
using StatsBase: kurtosis,norm

struct StateDensity
    parameter::AbstractRange
    bifurcations::AbstractArray
end

function infer( f::Function, J::Function, u₀::Vector{T}, θ::AbstractArray, data::StateDensity;
        optimiser=Momentum(), progress::Function=()->(),
        iter::Int=100, maxSteps::Int=1000, maxIter::Int=1000, tol=1e-10 ) where T

    parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=maxSteps,

			newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
			verbose=false,maxIter=maxIter,tol=tol),

		computeEigenValues = false)

    function predictor()
        bifurcations,u₀ = continuation( f,J,u₀, parameters )
        return bifurcations
    end

    function loss()
        bifurcations = predictor()

        if length(bifurcations.bifpoint) == 0
            return exp(-kurtosis( bifurcations.branch[1,:] ))

        elseif length(bifurcations.bifpoint) == 2
            return norm( data.bifurcations .- map( point -> point.param, bifurcations.bifpoint) )

    	else
    		return throw("unhandled bifurcations occured!")
        end
    end

    @time Flux.train!(loss, Zygote.Params([θ]), iter, optimiser, cb=progress)
end

function predictor()
    global u₀
	bifurcations,u₀ = continuation( f,J,u₀, parameters )
	return bifurcations
end

function loss()
    bifurcations = predictor()

    if length(bifurcations.bifpoint) == 0
        return exp(-kurtosis( bifurcations.branch[1,:] ))

    elseif length(bifurcations.bifpoint) == 2
        return norm( data.bifurcations .- map( point -> point.param, bifurcations.bifpoint) )

	else
		return throw("unhandled bifurcations occured!")
    end
end

function progress()
    bifurcations = predictor()
    prediction = kde( bifurcations.branch[1,:], data.parameter, bandwidth=2*parameters.ds)

    plotBranch( bifurcations, label="inferred")
    plot!( prediction.x, prediction.density, label="inferred", color="darkblue")
    plot!( data.bifurcations, zeros(length(data.bifurcations)), linewidth=3, label="target", color="gold", xlabel="parameter, p", ylabel="steady state") |> display
end

function lossAt(params...)
	copyto!(θ,[params...])
	try
		return loss()
	catch
		return NaN
	end
end
