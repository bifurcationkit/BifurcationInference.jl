module FluxContinuation

	using BifurcationKit: PALCIterable, newton, ContinuationPar, NewtonPar, DeflationOperator
	using BifurcationKit: BorderedArray, AbstractLinearSolver, AbstractEigenSolver, BorderingBLS
	using BifurcationKit: PALCStateVariables, solution, computeEigenvalues!

	using ForwardDiff: jacobian,gradient,hessian
	using Flux: Momentum,update!

	using Setfield: @lens,@set,setproperties
	using Parameters: @unpack

	using InvertedIndices: Not
	using LinearAlgebra, StaticArrays

	using Plots.PlotMeasures
	using LaTeXStrings
	using Plots

	include("Structures.jl")
	include("Gradients.jl")
	include("Utils.jl")

	export StateSpace,deflationContinuation,train!
	export getParameters,loss,∇loss
	export plot,@unpack

	""" root finding with newton deflation method"""
	function findRoots!( f::Function, J::Function, roots::AbstractVector{<:AbstractVector},
		parameters::NamedTuple, hyperparameters::ContinuationPar;
		maxRoots::Int = 3, maxIter::Int=500, verbosity=0 )

		hyperparameters = @set hyperparameters.newtonOptions = setproperties(
			hyperparameters.newtonOptions; maxIter = maxIter, verbose = verbosity )

		# search for roots across parameter range
		pRange = range(hyperparameters.pMin,hyperparameters.pMax,length=length(roots))
		roots .= findRoots.( Ref(f), Ref(J), roots, pRange, Ref(parameters), Ref(hyperparameters); maxRoots=maxRoots )
	end

	function findRoots( f::Function, J::Function, roots::AbstractVector{V}, p::T,
		parameters::NamedTuple, hyperparameters::ContinuationPar{T, S, E}; maxRoots::Int = 3, converged = false
		) where { T<:Number, V<:AbstractVector{T}, S<:AbstractLinearSolver, E<:AbstractEigenSolver }

		Zero = zero(first(roots))
		inf = Zero .+ Inf

		# search for roots at specific parameter value
		deflation = DeflationOperator(one(T), dot, one(T), [inf] ) # dummy deflation at infinity
		parameters = @set parameters.p = p

        for u ∈ roots # update existing roots
    		u, residual, converged, niter = newton( f, J, u.+hyperparameters.ds, parameters,
				hyperparameters.newtonOptions, deflation)

			@assert( !any(isnan.(residual)), "f(u,p) = $(residual[end]) at u = $u, p = $(parameters.p), θ = $(parameters.θ)")
    		if converged push!(deflation,u) else break end
        end

		u = Zero
		if converged || length(deflation)==1 # search for new roots
			while length(deflation)-1 < maxRoots

				u, residual, converged, niter = newton( f, J, u.+hyperparameters.ds, parameters,
					hyperparameters.newtonOptions, deflation)

				# make sure new roots are different from existing
				if any( isapprox.( Ref(u), deflation.roots, atol=2*hyperparameters.ds ) ) break end
				if converged push!(deflation,u) else break end
			end
		end

		filter!( root->root≠inf, deflation.roots ) # remove dummy deflation at infinity
		@assert( length(deflation.roots)>0, "No roots f(u,p)=0 found at p = $(parameters.p), θ = $(parameters.θ); try increasing maxIter")
		return deflation.roots
	end

	""" deflation continuation method """
	function deflationContinuation( f::Function, roots::AbstractVector{<:AbstractVector{V}},
		parameters::NamedTuple, hyperparameters::ContinuationPar{T, S, E};
		maxRoots::Int = 3, maxIter::Int=500, resolution=400, verbosity=0, kwargs...
		) where {T<:Number, V<:AbstractVector{T}, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		maxIterContinuation,ds = hyperparameters.newtonOptions.maxIter,hyperparameters.ds
		J(u,p) = jacobian(x->f(x,p),u)

		findRoots!( f, J, roots, parameters, hyperparameters; maxRoots=maxRoots, maxIter=maxIter, verbosity=verbosity)
		pRange = range(hyperparameters.pMin,hyperparameters.pMax,length=length(roots))
	    intervals = ([zero(T),step(pRange)],[-step(pRange),zero(T)])

		branches = Branch{V,T}[]
		hyperparameters = @set hyperparameters.newtonOptions.maxIter = maxIterContinuation
		linsolver = BorderingBLS(hyperparameters.newtonOptions.linsolver)

	    for (i,us) ∈ enumerate(roots)
	        for u ∈ us # perform continuation for each root

	            # forwards and backwards branches
	            for (pMin,pMax) ∈ intervals

					hyperparameters = setproperties(hyperparameters;
						pMin=pRange[i]+pMin, pMax=pRange[i]+pMax,
						ds=sign(hyperparameters.ds)*ds)

					# only store branches within observation range
					if hyperparameters.pMax <= maximum(pRange) && hyperparameters.pMin >= minimum(pRange)

		                # main continuation method
						branch = Branch(V,T)
						parameters = @set parameters.p = pRange[i]+hyperparameters.ds

						try
							iterator = PALCIterable( f, J, u, parameters, (@lens _.p), hyperparameters, linsolver, verbosity=verbosity)
							for state ∈ iterator

								computeEigenvalues!(iterator,state)
								push!(branch,state)
							end
							push!(branches,branch)

						catch error
							printstyled(color=:red,"Continuation Error at f(u,p)=$(f(u,parameters))\nu=$u, p=$(parameters.p), θ=$(parameters.θ)\n")
							rethrow(error)
						end
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
