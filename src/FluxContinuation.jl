module FluxContinuation

	using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig, AbstractLinearSolver, AbstractEigenSolver, DeflationOperator
	using PseudoArcLengthContinuation; const Cont = PseudoArcLengthContinuation

	using Zygote: Buffer, @nograd
	using StatsBase: norm
	using LinearAlgebra: dot
	using SliceMap: maprows

	using Plots,Printf
	using Plots.PlotMeasures
	using LaTeXStrings

	using Dates: now
	@nograd now,string

	include("patches/LinearAlgebra.jl")
	include("patches/Flux.jl")

	export StateDensity,deflationContinuation,findRoots
	export getParameters,updateParameters,loss

	struct StateDensity
	    parameter::AbstractRange
	    bifurcations::AbstractArray
	end

	""" root finding with newton deflation method"""
	function findRoots( f::Function, J::Function, u₀::Vector{Array{T,2}},
		parameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500,
			) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		pDeflations = range(parameters.pMin,parameters.pMax,length=length(u₀))
		rootsArray = similar(u₀)

		parameters.newtonOptions.maxIter = maxIter
	    for (i,us) in enumerate(u₀)

			deflation = DeflationOperator(1.0, dot, 1.0, [])
			converged = true

	        for u in eachrow(us) # update existing roots
	    		u, _, converged, _ = Cont.newtonDeflated( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
		    		u.+parameters.ds, parameters.newtonOptions, deflation)
	    		if converged push!(deflation,u) else break end
	        end

			if converged # search for new roots
				u = fill(0.0,nStates)

				while length(deflation) < maxRoots
					u, _, converged, _ = Cont.newtonDeflated( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
						u.+parameters.ds, parameters.newtonOptions, deflation)
					if converged & all(maprows( v -> !isapprox(u,v,atol=2*parameters.ds), deflation.roots[1:deflation.n_roots,:] ))
						push!(deflation,u)
					else break end
				end
			end
			rootsArray[i] = deflation.roots
	    end

		return rootsArray,pDeflations
	end
	@nograd findRoots

	""" differentiable deflation continuation method """
	function deflationContinuation( f::Function, J::Function, u₀::Vector{Array{T,2}}, parameters::ContinuationPar{T, S, E},
		printsolution::Function = u->u[1], finaliseSolution::Function = (_,_,_,_) -> true, maxRoots::Int = 3, maxIter::Int=500,
			) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		branchParameters = deepcopy(parameters)
		nStates = size(u₀[1])[2]

		rootsArray,pDeflations = findRoots( f, J, u₀, branchParameters, maxRoots, maxIter )
	    intervals = ([0.0,step(pDeflations)],[-step(pDeflations),0.0])
	    n_roots = div.(length.(rootsArray),nStates)

		# continuation per root
		branches = Buffer([], 2*sum(n_roots)-n_roots[end]-n_roots[1] ); n = 0
		branchParameters.newtonOptions.maxIter = parameters.newtonOptions.maxIter

	    for (i,us) in enumerate(rootsArray)
	        for u in eachrow(us)

	            # forwards and backwards branches
	            for (pMin,pMax) in intervals
	        		branchParameters.pMin,branchParameters.pMax = pDeflations[i]+pMin, pDeflations[i]+pMax
					branchParameters.ds = sign(branchParameters.ds)*copy(parameters.ds)

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

		return copy(branches), rootsArray
	end

	############################################# hyperparameter update and inference loop
	function getParameters(data::StateDensity; maxIter::Int=10, tol=1e-12 )
		return ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(

	        pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
	        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),

				newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
				verbose=false,maxIter=maxIter,tol=tol),

			detect_fold = false, detect_bifurcation = true)
	end
	@nograd getParameters

	function updateParameters(parameters::ContinuationPar{T, S, E}, steady_states; resolution=200
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		# estimate scale from steady state curves
		branch_points = map(length,steady_states)
		parameters.ds = parameters.dsmax = parameters.dsmin = maximum(branch_points)*parameters.ds/resolution

		return parameters
	end
	@nograd updateParameters

	########################################################### objective function
	function loss(steady_states, data::StateDensity, curvature::Function)

	    predictions = map( branch->map( point->point.param, branch.bifpoint),
	        filter( branch->length(branch.bifpoint) > 0, steady_states ))

	    # supervised signal
		predictions = vcat(predictions...)
	    error = length(predictions) > 0 ? norm(minimum(abs.(data.bifurcations.-predictions'),dims=1)) : Inf

	    # unsupervised signal
	    weight = length(predictions)!=2length(data.bifurcations)
	    curvature = sum( branch-> sum(
			abs.(curvature(branch.branch[2,:],branch.branch[1,:])).*abs.(branch.branch[4,:])),
			steady_states )

		return (1-weight)*tanh(log(error)) - weight*log(1+curvature)
	end

	############################################################## plotting
	import Plots: plot
	function plot(steady_states, data::StateDensity)

	    vline( data.bifurcations, label="", color=:gold, xlabel=L"\mathrm{parameter},\,p",
			right_margin=20mm,size=(500,400)); right_axis = twinx()

	    for branch in steady_states
			bifpt = branch.bifpoint[1:branch.n_bifurcations]

	        plot!(branch.branch[1,:],branch.branch[2,:], alpha=0.5, label="", grid=false, ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0",
	            color=map(x -> isodd(x) ? :darkblue : :lightblue, branch.stability[1:branch.n_points]), linewidth=2)

			determinant = map( x -> ( (eigenvalues,vectors,i) = x; prod(eigenvalues) ), branch.eig)
	        plot!(right_axis,branch.branch[1,:], determinant, alpha=0.5, label="", grid=false, ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(z)",
	            colour=map(x -> isodd(x) ? :red : :pink, branch.stability[1:branch.n_points]), linewidth=2)

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
	@nograd plot
end
