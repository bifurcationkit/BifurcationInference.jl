module FluxContinuation

	using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig, AbstractLinearSolver, AbstractEigenSolver, DeflationOperator
	using PseudoArcLengthContinuation; const Cont = PseudoArcLengthContinuation

	include("patches/PseudoArcLengthContinuation.jl")
	include("patches/LinearAlgebra.jl")
	include("patches/Flux.jl")
	export continuation,deflationContinuation

	""" extension of co-dimension one parameter continuation methods work with Zygote """
	function continuation( f::Function, J::Function, u₀::Vector{T}, parameters::ContinuationPar{T, S, E},
		printsolution::Function = u->u[1], finaliseSolution::Function = (_,_,_,_) -> true
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		function update( state::BorderedArray, gradient::BorderedArray, step::Int64, contResult::ContResult )
			if step == 0 u₀ = state.u end
			return finaliseSolution(state,gradient,step,contResult)
		end

		p₀ = parameters.pMin+parameters.ds
		bifurcations, = Cont.continuation( f, J, u₀, p₀, parameters;
			printsolution = printsolution, finaliseSolution = update )

		return bifurcations, u₀

	end

	""" extension of deflation continuation method """
	function deflationContinuation( f::Function, J::Function, u₀::Vector{Array{T,2}}, parameters::ContinuationPar{T, S, E},
		printsolution::Function = u->u[1], finaliseSolution::Function = (_,_,_,_) -> true, maxRoots::Int = 3, maxIter::Int=500,
			) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		nDeflations,nStates = length(u₀),size(u₀[1])[2]
		default_root = fill(Inf,nStates)

		pDeflations = range(parameters.pMin,parameters.pMax,length=nDeflations)
		branchParameters = deepcopy(parameters)

		pStep = step(pDeflations)
	    intervals = ([0.0,pStep],[-pStep,0.0])

	    # find roots with deflated newton method
		rootsArray = Buffer([[1.0 1.0]], nDeflations )
		branchParameters.newtonOptions.maxIter = maxIter
	    for (i,us) in enumerate(u₀)

			roots = Buffer([Inf],maxRoots,nStates)
			for i=1:maxRoots roots[i,:] = default_root end

			deflation = DeflationOperator(1.0, dot, 1.0, roots, 0)
			converged = true

	        for u in eachrow(us) # update existing roots
	    		u, _, converged, _ = Cont.newtonDeflated( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
		    		u.+branchParameters.ds, branchParameters.newtonOptions, deflation)
	    		if converged push!(deflation,u) else break end
	        end

			if converged # search for new roots
				u = fill(0.0,nStates)

				while length(deflation) < maxRoots
					u, _, converged, _ = Cont.newtonDeflated( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
						u.+branchParameters.ds, branchParameters.newtonOptions, deflation)
					if converged & all([ !isapprox(u,v,atol=2*parameters.ds) for v in eachrow(deflation.roots[1:deflation.n_roots,:]) ])
						push!(deflation,u)
					else break end
				end
			end
			rootsArray[i] = copy(deflation.roots)[1:deflation.n_roots,:]
	    end

		rootsArray = copy(rootsArray)
	    n_roots = div.(length.(rootsArray),nStates)

		branches = Buffer([], 2*sum(n_roots)-n_roots[end]-n_roots[1] )
	    n = 0

	    # continuation per root
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
end
