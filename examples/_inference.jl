using PseudoArcLengthContinuation: plotBranch,ContinuationPar,NewtonPar,DefaultLS,DefaultEig
using FluxContinuation: continuation,deflationContinuation
using Flux: train!

using Zygote, Plots, Printf
using StatsBase: norm,mean,std
using Plots.PlotMeasures
using LaTeXStrings
gr()

struct StateDensity
    parameter::AbstractRange
    bifurcations::AbstractArray
end

############################################# hyperparameters and inference loop
function getParameters(data::StateDensity; maxIter::Int=10, tol=1e-12 )
	return ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(

        pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
        ds=step(data.parameter)/10, dsmax=step(data.parameter)/2, dsmin=step(data.parameter)/100,

			newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
			verbose=false,maxIter=maxIter,tol=tol),

		detect_fold = false, detect_bifurcation = true)
end

function infer( f::Function, J::Function, u₀::Vector{T}, θ::AbstractArray, data::StateDensity;
        optimiser=ADAM(0.05), progress::Function=()->(), iter::Int=100, maxIter::Int=10, tol=1e-12 ) where T

    parameters = getParameters(data; maxIter=maxIter, tol=tol)
    @time train!(loss, Zygote.Params([θ]), iter, optimiser, cb=progress)
end

############################################# steady state and bifurcation prediction
function predictor() global u₀
	bifurcations,u₀ = deflationContinuation( (u,p)->f(u,p), (u,p)->J(u,p), u₀, parameters )
	return bifurcations
end

############################################# determinant and its signed curvature
function Δ(bifurcations)
    determinant = map( x -> ( (eigenvalues,vectors,i) = x; prod(eigenvalues) ), bifurcations.eig)
	return bifurcations.branch[1,:],determinant
end

function Tr(bifurcations)
    trace = map( x -> ( (eigenvalues,vectors,i) = x; sum(eigenvalues) ), bifurcations.eig)
	return bifurcations.branch[1,:],trace
end

function κ(bifurcations)
	p,u,ds = bifurcations.branch[1,:], bifurcations.branch[2,:], bifurcations.branch[4,:]
    return K(u,p), abs.(ds)
end

############################################# objective function
function loss()

    steady_states = predictor()
    predictions = map( branch->map( point->point.param, branch.bifpoint),
        filter( branch->length(branch.bifpoint) > 0, steady_states ))

    # supervised signal
    predictions = collect(Iterators.flatten(predictions))
    error = length(predictions) > 0 ? norm(minimum(abs.(data.bifurcations.-predictions'),dims=1)) : Inf

    # unsupervised signal
    weight = length(predictions)!=2length(data.bifurcations)
    curvature = sum( branch->( (k,ds)=κ(branch); sum( abs.(k) .* ds ) ), steady_states )

    mean_stability = mean( branch->( (p,T)=Tr(branch); mean(T) ), steady_states )
    return (1-weight)*(tanh(log(error))-1) - weight*(log(1+curvature)-1) + tanh(mean_stability)
end

############################################# plotting function
function progress()
    bifurcations = predictor()

    vline( data.bifurcations, label="", color=:gold, xlabel=L"\mathrm{parameter},\,p",right_margin=20mm,size=(500,400))
	right_axis = twinx()

    for bifurcation in bifurcations
		bifpt = bifurcation.bifpoint[1:bifurcation.n_bifurcations]

        plot!(bifurcation.branch[1,:],bifurcation.branch[2,:], alpha=0.5, label="", grid=false, ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0",
            color=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]), linewidth=2)

        plot!(right_axis,Δ(bifurcation)..., alpha=0.5, label="", grid=false, ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(z)",
            colour=map(x -> isodd(x) ? :red : :pink, bifurcation.stability[1:bifurcation.n_points]), linewidth=2)

		scatter!(map(x->x.param,bifpt),map(x->x.printsol,bifpt), label="",
            m = (3.0, 3.0, :black, stroke(0, :none)))
    end

	plot!(right_axis,[],[], color=:gold, legend=:bottomleft,
        alpha=1.0, label=L"\mathrm{targets}\,\,\mathcal{D}")
	scatter!(right_axis,[],[], label=L"\mathrm{prediction}\,\,\mathcal{P}(\theta)", legend=:bottomleft,
		m = (1.0, 1.0, :black, stroke(0, :none)))
	plot!(right_axis,[],[], color=:darkblue, legend=:bottomleft, linewidth=2,
        alpha=1.0, label=L"\mathrm{steady\,states}")
	plot!(right_axis,[],[], color=:red, legend=:bottomleft,
        alpha=1.0, label=L"\mathrm{determinant}", dpi=500, linewidth=2) |> display
end

function lossAt(params...)
	copyto!(θ,[params...]); loss()
	return loss()
end
