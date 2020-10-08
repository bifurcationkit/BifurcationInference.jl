module FluxContinuation

	using BifurcationKit: PALCIterable, newton, ContinuationPar, NewtonPar, DeflationOperator
	using BifurcationKit: getx, getp, detectBifucation, computeEigenvalues!

	using LinearAlgebra: det,dot,eigen,norm,tr
	using ForwardDiff: jacobian,gradient,hessian
	using Flux: Momentum,update!

	using Setfield: @lens,@set,set,setproperties,Lens
	using Parameters: @unpack,@with_kw
	using InvertedIndices: Not

	using Plots.PlotMeasures
	using LaTeXStrings
	using Plots	

	include("Structures.jl")
	include("Gradients.jl")
	include("Utils.jl")

	export StateDensity,deflationContinuation,train!
	export getParameters,loss,∇loss
	export plot,@unpack,@lens,set

	""" root finding with newton deflation method"""
	function findRoots!( f::Function, J::Function, roots::Vector{Vector{Vector{T}}}, params::NamedTuple, paramlens::Lens,
		hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500; verbosity=0
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		nStates = length(first(first(roots)))
		rootsArray = similar(roots)

		pDeflations = range(hyperparameters.pMin,hyperparameters.pMax,length=length(roots))
		hyperparameters = @set hyperparameters.newtonOptions = setproperties(
			hyperparameters.newtonOptions; maxIter = maxIter, verbose = verbosity )

	    for (i,us) ∈ enumerate(roots)

			deflation = DeflationOperator(T(1.0), dot, T(1.0), [fill(T(Inf),nStates)] ) # initialise dummy deflation at ∞
			converged = true

	        for u ∈ us # update existing roots
	    		u, residual, converged, niter = newton( f, J, u.+hyperparameters.ds, set(params, paramlens, pDeflations[i]),
					hyperparameters.newtonOptions, deflation)

				if any(isnan.(residual)) throw("f(u,p) = NaN, u = $u, p = $(set(params, paramlens, pDeflations[i]))") end
	    		if converged push!(deflation,u) else break end
	        end

			if converged # search for new roots
				u = zeros(T,nStates)
				while length(deflation)-1 < maxRoots

					u, residual, converged, niter = newton( f, J, u.+hyperparameters.ds, set(params, paramlens, pDeflations[i]),
						hyperparameters.newtonOptions, deflation)

					if converged & all(map( v -> !isapprox(u,v,atol=2*hyperparameters.ds), deflation.roots))
						push!(deflation,u)
					else break end
				end
			end

			filter!(root->root≠fill(T(Inf),nStates),deflation.roots) # remove dummy deflation at ∞
			rootsArray[i] = deflation.roots
		end

		# in-place update with new roots
		roots .= rootsArray
	end

	""" differentiable deflation continuation method """
	function deflationContinuation( f::Function, roots::Vector{Vector{Vector{T}}}, params::NamedTuple, paramlens::Lens,
		hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500, resolution=400; verbosity=0
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		maxIterContinuation,ds = hyperparameters.newtonOptions.maxIter,hyperparameters.ds
		J(u,p) = jacobian(x->f(x,p),u)

		findRoots!( f, J, roots, params, paramlens, hyperparameters, maxRoots, maxIter; verbosity=verbosity )
		pDeflations = range(hyperparameters.pMin,hyperparameters.pMax,length=length(roots))
	    intervals = ([T(0.0),step(pDeflations)],[-step(pDeflations),T(0.0)])

		# continuation per root
		branches = Branch{T}[]
		linsolver = BorderedLinearSolver()
		hyperparameters = @set hyperparameters.newtonOptions.maxIter = maxIterContinuation

	    for (i,us) ∈ enumerate(roots)
	        for u ∈ us

	            # forwards and backwards branches
	            for (pMin,pMax) in intervals

					hyperparameters = setproperties(hyperparameters;
						pMin=pDeflations[i]+pMin, pMax=pDeflations[i]+pMax,
						ds=sign(hyperparameters.ds)*ds)

	                # main continuation method
					branch = Branch(T)
					params = set(params, paramlens, pDeflations[i]+hyperparameters.ds)
					iterator = PALCIterable( f, J, u, params, paramlens, hyperparameters, linsolver, verbosity=verbosity)

					for state in iterator

						push!(branch.state,copy(getx(state)))
						push!(branch.parameter,copy(getp(state)))
						push!(branch.ds,state.ds)

						computeEigenvalues!(iterator,state)
						push!(branch.eigvals,state.eigvals)
						push!(branch.bifurcations,detectBifucation(state))
					end

	        		if hyperparameters.pMax <= maximum(pDeflations) && hyperparameters.pMin >= minimum(pDeflations)
	        			push!(branches,branch) end
	        		hyperparameters = @set hyperparameters.ds = -hyperparameters.ds
	            end
	    	end
	    end

		hyperparameters = @set hyperparameters.ds = ds
		updateParameters!(hyperparameters,branches;resolution=resolution)
		return branches
	end
end
