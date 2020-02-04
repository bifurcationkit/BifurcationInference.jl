using PseudoArcLengthContinuation: plotBranch,ContinuationPar,NewtonPar,DefaultLS,DefaultEig
using FluxContinuation: continuation,deflationContinuation
using Flux: train!

using Zygote, Plots, Printf
using StatsBase: kurtosis,norm,mean,std

struct StateDensity
    parameter::AbstractRange
    bifurcations::AbstractArray
end

function infer( f::Function, J::Function, u₀::Vector{T}, θ::AbstractArray, data::StateDensity;
        optimiser=ADAM(0.05), progress::Function=()->(),
        iter::Int=100, maxSteps::Int=1000, maxIter::Int=3000, tol=1e-10 ) where T

    parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		pMin=minimum(data.parameter),pMax=maximum(data.parameter),ds=step(data.parameter), maxSteps=maxSteps,

			newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
			verbose=false,maxIter=maxIter,tol=tol),

		detect_fold = false, detect_bifurcation = true)

    @time train!(loss, Zygote.Params([θ]), iter, optimiser, cb=progress)
end

function predictor() global u₀
	bifurcations,u₀ = deflationContinuation( f,J,u₀, parameters )
	return bifurcations
end

function potential(bifurcations)

	p,v = det(bifurcations)
	ds = bifurcations.branch[4,:]
    v = abs.(v)

	dp = (p[1:end-2]-2*p[2:end-1]+p[3:end]) ./ ds[2:end-1]
	dv = (v[1:end-2]-2*v[2:end-1]+v[3:end]) ./ ds[2:end-1]

    curvature = sqrt.(dp.^2+dv.^2)[3:end-2]
    parameter = p[4:end-3]

    return parameter,curvature
end

function det(bifurcations)
    determinant = map( x -> ( (eigenvalues,vectors,i) = x; prod(eigenvalues) ), bifurcations.eig)
	return bifurcations.branch[1,:],determinant
end

function loss()
    steady_states = predictor()

    if sum( branch -> length(branch.bifpoint), steady_states) == 0
		return -log(sum(vcat(map(branch->((p,v)=potential(branch); v), steady_states)...))*step(data.parameter))

    else
        bifurcations = steady_states[map( branch -> length(branch.bifpoint) > 0, steady_states)]
        points = vcat(map(branch -> map( point -> point.param, branch.bifpoint), bifurcations)...)
        return norm(data.bifurcations.-points') .- log(norm(points.-points')+1)
    end
end

function progress()
    bifurcations = predictor()
    plot( data.bifurcations, zeros(length(data.bifurcations)), label="", color="gold", xlabel="parameter, p")
	right_axis = twinx()

    for bifurcation in bifurcations

        plot!(bifurcation.branch[1,:],bifurcation.branch[2,:], alpha=0.5, label="", grid=false, ylabel="steady states",
            color=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]),
			markersize=2, markercolor=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]),
			markershape=:circle, markerstrokewidth=0)

        bifpt = bifurcation.bifpoint[1:bifurcation.n_bifurcations]
        scatter!(map(x->x.param,bifpt),map(x->x.printsol,bifpt), label="",
            color = :black, alpha=0.5, markersize=3, markerstrokewidth=0)

        plot!(right_axis,det(bifurcation)..., alpha=0.5, label="", grid=false, ylabel="determinant",
            colour=map(x -> isodd(x) ? :red : :pink, bifurcation.stability[1:bifurcation.n_points]))
    end

	p = range(minimum(parameter),maximum(parameter),length=length(u₀))
    for (i,us) in enumerate(u₀)
		for u in eachrow(us)
			plot!([p[i]],u,markersize=2,marker=:circle,label="",color=:darkblue)
		end
	end

    plot!(right_axis,[],[], color=:red, legend=:bottomright,
        alpha=0.5, label="determinant")
    plot!([],[], color=:darkblue, legend=:bottomleft,
        alpha=0.5, label="steady states")  |> display
end

function lossAt(params...)
	try
		copyto!(θ,[params...])
		return loss()
	catch
		return NaN
	end
end
