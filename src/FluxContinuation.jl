module FluxContinuation

	using BifurcationKit: PALCIterable, newton, ContinuationPar, NewtonPar, DeflationOperator
	using BifurcationKit: BorderedArray, AbstractLinearSolver, AbstractEigenSolver
	using BifurcationKit: PALCStateVariables, solution, computeEigenvalues!

	using ForwardDiff: jacobian,gradient,hessian
	using Flux: Momentum,update!

	using Setfield: @lens,@set,set,setproperties,Lens
	using Parameters: @unpack

	using InvertedIndices: Not
	using LinearAlgebra
	include("patches/LinearAlgebra.jl")

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
	function findRoots!( f::Function, J::Function, roots::AbstractVector{<:AbstractVector},
		params::NamedTuple, paramlens::Lens, hyperparameters::ContinuationPar,
		maxRoots::Int = 3, maxIter::Int=500; verbosity=0 )

		hyperparameters = @set hyperparameters.newtonOptions = setproperties(
			hyperparameters.newtonOptions; maxIter = maxIter, verbose = verbosity )

		pRange = range(hyperparameters.pMin,hyperparameters.pMax,length=length(roots))
		roots .= findRoots.( Ref(f), Ref(J), roots, pRange, Ref(params), Ref(paramlens), Ref(hyperparameters), Ref(maxRoots) )
	end

	function findRoots( f::Function, J::Function, roots::AbstractVector{V}, p::T,
		params::NamedTuple, paramlens::Lens, hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3
		) where { T<:Number, V<:AbstractVector{T}, S<:AbstractLinearSolver, E<:AbstractEigenSolver }

		inf = convert(V, fill(Inf,length(first(roots))) )
		deflation = DeflationOperator(T(1.0), dot, T(1.0), [inf] ) # initialise dummy deflation at ∞
		converged = true

        for u ∈ roots # update existing roots
    		u, residual, converged, niter = newton( f, J, u.+hyperparameters.ds, set(params, paramlens, p),
				hyperparameters.newtonOptions, deflation)

			if any(isnan.(residual)) throw("f(u,p) = NaN, u = $u, p = $(set(params, paramlens, p))") end
    		if converged push!(deflation,u) else break end
        end

		u = convert(V, fill(0,length(first(roots))) )
		if converged # search for new roots
			while length(deflation)-1 < maxRoots

				u, residual, converged, niter = newton( f, J, u.+hyperparameters.ds, set(params, paramlens, p),
					hyperparameters.newtonOptions, deflation)

				if any( isapprox.( Ref(u), deflation.roots, atol=2*hyperparameters.ds ) ) break end
				if converged push!(deflation,u) else break end
			end
		end

		filter!( root->root≠inf, deflation.roots ) # remove dummy deflation at ∞
		return deflation.roots
	end

	""" deflation continuation method """
	function deflationContinuation( f::Function, roots::AbstractVector{<:AbstractVector{V}},
		params::NamedTuple, paramlens::Lens, hyperparameters::ContinuationPar{T, S, E},
		maxRoots::Int = 3, maxIter::Int=500, resolution=400; verbosity=0
		) where {T<:Number, V<:AbstractVector{T}, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		maxIterContinuation,ds = hyperparameters.newtonOptions.maxIter,hyperparameters.ds
		J(u,p) = jacobian(x->f(x,p),u)

		findRoots!( f, J, roots, params, paramlens, hyperparameters, maxRoots, maxIter; verbosity=verbosity )
		pRange = range(hyperparameters.pMin,hyperparameters.pMax,length=length(roots))
	    intervals = ([T(0.0),step(pRange)],[-step(pRange),T(0.0)])

		# continuation per root
		branches = Branch{V,T}[]
		hyperparameters = @set hyperparameters.newtonOptions.maxIter = maxIterContinuation

	    for (i,us) ∈ enumerate(roots)
	        for u ∈ us

	            # forwards and backwards branches
	            for (pMin,pMax) ∈ intervals

					hyperparameters = setproperties(hyperparameters;
						pMin=pRange[i]+pMin, pMax=pRange[i]+pMax,
						ds=sign(hyperparameters.ds)*ds)

					if hyperparameters.pMax <= maximum(pRange) && hyperparameters.pMin >= minimum(pRange)

		                # main continuation method
						branch = Branch(V,T)
						params = set(params, paramlens, pRange[i]+hyperparameters.ds)

						iterator = PALCIterable( f, J, u, params, paramlens, hyperparameters, verbosity=verbosity)
						for state ∈ iterator

							computeEigenvalues!(iterator,state)
							push!(branch,state)
						end

		        		push!(branches,branch)
		        		hyperparameters = @set hyperparameters.ds = -hyperparameters.ds
					end
	            end
	    	end
	    end

		hyperparameters = @set hyperparameters.ds = ds
		updateParameters!(hyperparameters,branches;resolution=resolution)
		return branches
	end
end
