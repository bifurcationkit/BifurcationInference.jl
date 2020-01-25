include("../src/patches/KernelDensity.jl")
using PseudoArcLengthContinuation: plotBranch,ContinuationPar,NewtonPar,DefaultLS,DefaultEig
using FluxContinuation: continuation,deflationContinuation

using Flux, Zygote, Plots, Printf
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

    @time Flux.train!(loss, Zygote.Params([θ]), iter, optimiser, cb=progress)
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
        return -log(sum(vcat(map(branch->potential(branch)[2], steady_states)...))*step(data.parameter))

    else
        bifurcations = steady_states[map( branch -> length(branch.bifpoint) > 0, steady_states)]
        points = vcat(map(branch -> map( point -> point.param, branch.bifpoint), bifurcations)...)
        return norm(data.bifurcations.-points') .- log(norm(points.-points')+1)
    end
end

function progress()
    bifurcations = predictor()
    plot( data.bifurcations, zeros(length(data.bifurcations)), linewidth=3, label="target", color="gold", xlabel="parameter, p")

    μ = mean(vcat(map( branch -> branch.branch[2,:] , bifurcations )...))
    σ = std(vcat(map( branch -> branch.branch[2,:] , bifurcations )...))
    S = std(vcat(map( branch -> det(branch)[2] , bifurcations )...))

    for bifurcation in bifurcations
        p,v = det(bifurcation)
        plot!(p,v/S,linewidth=3, alpha=0.5, label="",
            colour=map(x -> isodd(x) ? :red : :pink, bifurcation.stability[1:bifurcation.n_points]))

        plot!(bifurcation.branch[1,:],(bifurcation.branch[2,:].-μ)/σ, linewidth=3, alpha=0.5, label="",
            color=map(x -> isodd(x) ? :darkblue : :lightblue, bifurcation.stability[1:bifurcation.n_points]))

        bifpt = bifurcation.bifpoint[1:bifurcation.n_bifurcations]
        scatter!(map(x->x.param,bifpt),(map(x->x.printsol,bifpt).-μ)/σ, label="",
            color = :black, alpha=0.5, markersize=5, markerstrokewidth=0)
    end

    plot!([],[], linewidth=3, color=:red,
        alpha=0.5, label="determinant")
    plot!([],[], linewidth=3, color=:darkblue, legend=:bottomleft,
        alpha=0.5, label="steady states")  |> display
end

function lossAt(params...)
	copyto!(θ,[params...])
    try
        return loss()
    catch
        return NaN
    end
end

import FluxContinuation: deflationContinuation
using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig, AbstractLinearSolver, AbstractEigenSolver, DeflationOperator
using PseudoArcLengthContinuation; const Cont = PseudoArcLengthContinuation
using LinearAlgebra: dot

""" extension of deflation continuation method """
function deflationContinuation( f::Function, J::Function, u₀::Vector{Array{T,2}}, parameters::ContinuationPar{T, S, E},
	printsolution::Function = u->u[1], finaliseSolution::Function = (_,_,_,_) -> true, maxRoots::Int = 3
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

    nDeflations = length(u₀)
    _,nStates = size(u₀[1])

	branchParameters = deepcopy(parameters)
	pDeflations = range(parameters.pMin,parameters.pMax,length=nDeflations)

	pStep = step(pDeflations)
    intervals = ([0.0,pStep],[-pStep,0.0])

    rootsArray = Vector{Array{Float64,2}}(undef,nDeflations)
    default_root = fill(Inf,nStates)

    # find roots with deflated newton method
    for (i,us) in enumerate(u₀)
        for u in eachrow(us)

    		roots = Buffer([Inf],maxRoots,nStates)
            for i=1:maxRoots roots[i,:] = default_root end
    		deflation = DeflationOperator(1.0, dot, 1.0, roots, 0)

    		while length(deflation) < maxRoots # search for roots
    			u, _, converged, _ = Cont.newtonDeflated( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
    				u.+5*abs(branchParameters.ds), branchParameters.newtonOptions, deflation)
    			if converged push!(deflation,u) else break end
    		end

            rootsArray[i] = copy(deflation.roots)[1:deflation.n_roots,:]
        end
    end

    n_roots = div.(length.(rootsArray),nStates)
    branches = Array{ContResult}(undef, 2*sum(n_roots)-n_roots[end]-n_roots[1] )
    n = 0

    # continuation per root
    for (i,us) in enumerate(rootsArray)
        for u in eachrow(us)

            # forwards and backwards branches
            for (pMin,pMax) in intervals
        		branchParameters.pMin,branchParameters.pMax = pDeflations[i]+pMin, pDeflations[i]+pMax

                # main continuation method
        		branch, = Cont.continuation( f, J, u, pDeflations[i]+branchParameters.ds,
        			branchParameters; printsolution = printsolution,
                    finaliseSolution = finaliseSolution)

        		if branchParameters.pMax <= maximum(pDeflations) && branchParameters.pMin >= minimum(pDeflations)
        			n +=1; branches[n] = branch end
        		branchParameters.ds *= -1.0
            end
    	end
    end

	return branches, u₀
end

θ = [0.1,-1.0]
u₀ = [[0.0 0.0], [0.0 0.0], [0.0 0.0; 1.0 1.0]]

progress()
plot(x, x-> lossAt(x,2.0) )
