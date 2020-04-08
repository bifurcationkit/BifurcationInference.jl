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

function infer( f::Function, J::Function, u₀::Vector{T}, θ::AbstractArray, data::StateDensity;
        optimiser=ADAM(0.05), progress::Function=()->(), iter::Int=100, maxIter::Int=10, tol=1e-12 ) where T

    parameters = getParameters(data; maxIter=maxIter, tol=tol)
    @time train!(loss, Zygote.Params([θ]), iter, optimiser, cb=progress)
end

function getParameters(data::StateDensity; maxIter::Int=10, tol=1e-12 )
	return ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(

        pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),

			newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
			verbose=false,maxIter=maxIter,tol=tol),

		detect_fold = false, detect_bifurcation = true)
end

function predictor() global u₀,A
	bifurcations,u₀ = deflationContinuation( (u,p)->f(A*u,p), (u,p)->A*J(A*u,p), u₀, parameters )
	A *= maximum(abs.(vcat(map( branch -> branch.branch[2,:], bifurcations)...)))
	return bifurcations
end

function potential(bifurcations)
	p,u = bifurcations.branch[1,:], bifurcations.branch[2,:]
    return p, K(p,u)
end

function det(bifurcations)
    determinant = map( x -> ( (eigenvalues,vectors,i) = x; prod(eigenvalues) ), bifurcations.eig)
	return bifurcations.branch[1,:],determinant
end

function loss()
    steady_states = predictor()
	return sum( branch->( (p,v)=potential(branch);
		sum(v)*step(data.parameter) ), steady_states )

    if sum( branch -> length(branch.bifpoint), steady_states) == 0
		return -log(1+sum( branch->( (p,v)=potential(branch);
			sum(v)*step(data.parameter) ), steady_states ))

    else
        bifurcations = steady_states[map( branch -> length(branch.bifpoint) > 0, steady_states)]
        points = vcat(map(branch -> map( point -> point.param, branch.bifpoint), bifurcations)...)
        return norm(data.bifurcations.-points') .- log(norm(points.-points'))
    end
end

function progress(states_only=false)
    bifurcations = predictor()

	if states_only
		for bifurcation in bifurcations
			bifpt = bifurcation.bifpoint[1:bifurcation.n_bifurcations]
			plot!(bifurcation.branch[1,:],bifurcation.branch[2,:], alpha=0.5, label="", grid=false, ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0",
				color=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]), linewidth=2)
		end
	else

	    vline( data.bifurcations, label="", color=:gold, xlabel=L"\mathrm{parameter},\,p",right_margin=20mm,size=(500,400))
		right_axis = twinx()

	    for bifurcation in bifurcations
			bifpt = bifurcation.bifpoint[1:bifurcation.n_bifurcations]

	        plot!(bifurcation.branch[1,:],bifurcation.branch[2,:], alpha=0.5, label="", grid=false, ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0",
	            color=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]), linewidth=2)

	        plot!(right_axis,det(bifurcation)..., alpha=0.5, label="", grid=false, ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(z)",
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
end

function lossAt(params...)
	copyto!(θ,[params...])
	loss()
	return loss()
end
