module FluxContinuation

	using PseudoArcLengthContinuation
	using PseudoArcLengthContinuation: AbstractLinearSolver, AbstractBorderedLinearSolver, AbstractEigenSolver, _axpy, detectBifucation, computeEigenvalues!
	using Zygote: Buffer, @nograd, @adjoint, @adjoint!

	using Setfield: @set,set,setproperties,Lens,@lens
	using Parameters: @with_kw
	using Dates: now

	using LinearAlgebra: dot,eigen
	using StatsBase: norm,mean

	using Plots.PlotMeasures
	using LaTeXStrings
	using Plots

	include("patches/PseudoArcLengthContinuation.jl")
	include("patches/Zygote.jl")

	include("Structures.jl")
	include("Utils.jl")

	export StateDensity,Branch,deflationContinuation,findRoots
	export getParameters,updateParameters,loss

	@nograd now,string

	""" root finding with newton deflation method"""
	function findRoots( f::Function, J::Function, u₀::Vector{Array{T,2}}, params::NamedTuple, paramlens::Lens,
		hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		pDeflations = range(hyperparameters.pMin,hyperparameters.pMax,length=length(u₀))
		rootsArray = similar(u₀)
		_,nStates = size(u₀[1])

		hyperparameters = @set hyperparameters.newtonOptions.maxIter = maxIter
	    for (i,us) in enumerate(u₀)
			params = set(params, paramlens, pDeflations[i])

			deflation = DeflationOperator(1.0, dot, 1.0, [fill(Inf,nStates)] )
			converged = true

	        for u in eachrow(us) # update existing roots
	    		u, _, converged, _ = newton( f, J, u.+hyperparameters.ds, params,
					hyperparameters.newtonOptions, deflation)
	    		if converged push!(deflation,u) else break end
	        end

			if converged # search for new roots
				u = fill(0.0,nStates)
				while length(deflation)-1 < maxRoots

					u, _, converged, _ = newton( f, J, u.+hyperparameters.ds, params,
						hyperparameters.newtonOptions, deflation)

					if converged & all(map( v -> !isapprox(u,v,atol=2*hyperparameters.ds), deflation.roots))
						push!(deflation,u)
					else break end
				end
			end

			filter!(root->root≠fill(Inf,nStates),deflation.roots)
			rootsArray[i] = transpose(hcat(deflation.roots...))
	    end

		return rootsArray,pDeflations
	end
	@nograd findRoots

	""" differentiable deflation continuation method """
	function deflationContinuation( f::Function, J::Function, u₀::Vector{Array{T,2}}, params::NamedTuple, paramlens::Lens,
		hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		maxIterContinuation,ds = hyperparameters.newtonOptions.maxIter,hyperparameters.ds
		rootsArray,pDeflations = findRoots( f, J, u₀, params, paramlens, hyperparameters, maxRoots, maxIter )
	    intervals = ([0.0,step(pDeflations)],[-step(pDeflations),0.0])

		# continuation per root
		branches = Buffer( Branch{T}[] )
		linsolver = BorderedLinearSolver()
		hyperparameters = @set hyperparameters.newtonOptions.maxIter = maxIterContinuation

	    for (i,us) in enumerate(rootsArray)
	        for u in eachrow(us)

	            # forwards and backwards branches
	            for (pMin,pMax) in intervals

					hyperparameters = setproperties(hyperparameters;
						pMin=pDeflations[i]+pMin, pMax=pDeflations[i]+pMax,
						ds=sign(hyperparameters.ds)*ds)

	                # main continuation method
					branch = BranchBuffer(T)
					params = set(params, paramlens, pDeflations[i]+hyperparameters.ds)
					iterator = PALCIterable( f, J, u, params, paramlens, hyperparameters, linsolver)

					for state in iterator

						push!(branch.state,copy(getx(state)))
						push!(branch.parameter,copy(getp(state)))
						push!(branch.ds,state.ds)

						computeEigenvalues!(iterator,state)
						push!(branch.eigvals,state.eigvals)
						push!(branch.bifurcations,detectBifucation(state))
					end

	        		if hyperparameters.pMax <= maximum(pDeflations) && hyperparameters.pMin >= minimum(pDeflations)
	        			push!(branches,copy(branch)) end
	        		hyperparameters = @set hyperparameters.ds = -hyperparameters.ds
	            end
	    	end
	    end

		hyperparameters = @set hyperparameters.ds = ds
		return copy(branches), rootsArray
	end

	""" semi-supervised objective function """
	function loss( steady_states::Vector{Branch{T}}, data::StateDensity{T}, determinant::Function, curvature::Function, parameters::NamedTuple;
		offset::Number=5.0, tol::Float64 = 0.1) where {T<:Number}

		# detect bifurcations
		parametrised_determinant(u,p) = determinant(u,p,parameters)
		weights = map( branch -> exp.(-parametrised_determinant.(branch.state,branch.parameter).^2), steady_states )

		# supervised signal
		errors = vcat(map( (weight,branch) -> minimum(weight .* abs.(branch.parameter.-data.bifurcations'), dims=2 ), weights,steady_states )...)
		supervised = log(1+norm(errors))

		parametrised_curvature(u,p) = curvature(u,p,parameters)
		K = sum( branch-> sum( parametrised_curvature.(branch.state,branch.parameter).^2 .* abs.(branch.ds) ), steady_states )
		unsupervised = -log(1+K)

		if sum( branch -> sum(branch.bifurcations), steady_states ) > 0
			return supervised - offset
		else
			return unsupervised
		end
	end
end
