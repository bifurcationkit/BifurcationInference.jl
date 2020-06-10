module FluxContinuation

	using PseudoArcLengthContinuation
	using PseudoArcLengthContinuation: AbstractLinearSolver, AbstractBorderedLinearSolver, AbstractEigenSolver, _axpy, detectBifucation, computeEigenvalues!
	using Zygote: Buffer, gradient, forward_jacobian, @nograd, @adjoint, @adjoint!

	using Setfield: @set,set,setproperties,Lens,@lens
	using Parameters: @with_kw, @unpack
	using Dates: now

	using LinearAlgebra: dot,eigen,svd
	using StatsBase: norm,mean
	using CuArrays

	using Plots.PlotMeasures
	using LaTeXStrings
	using Plots

	include("Structures.jl")
	include("Utils.jl")

	export StateDensity,Branch,deflationContinuation,findRoots
	export getParameters,updateParameters,loss,cu

	@nograd now,string

	""" root finding with newton deflation method"""
	function findRoots( f::Function, J::Function, u₀::Vector{Array{T,2}}, params::NamedTuple, paramlens::Lens,
		hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500; verbosity=0
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		_,nStates = size(first(u₀))
		rootsArray = similar(u₀)

		pDeflations = range(hyperparameters.pMin,hyperparameters.pMax,length=length(u₀))
		hyperparameters = @set hyperparameters.newtonOptions = setproperties(
			hyperparameters.newtonOptions; maxIter = maxIter, verbose = verbosity )

	    for (i,us) in enumerate(u₀)

			deflation = DeflationOperator(T(1.0), dot, T(1.0), [fill(T(Inf),nStates)] )
			converged = true

	        for u in eachrow(us) # update existing roots
	    		u, _, converged, _ = newton( f, J, u.+hyperparameters.ds, set(params, paramlens, pDeflations[i]),
					hyperparameters.newtonOptions, deflation)
	    		if converged push!(deflation,u) else break end
	        end

			if converged # search for new roots
				u = fill(T(0.0),nStates)
				while length(deflation)-1 < maxRoots

					u, _, converged, _ = newton( f, J, u.+hyperparameters.ds, set(params, paramlens, pDeflations[i]),
						hyperparameters.newtonOptions, deflation)

					if converged & all(map( v -> !isapprox(u,v,atol=2*hyperparameters.ds), deflation.roots))
						push!(deflation,u)
					else break end
				end
			end

			filter!(root->root≠fill(T(Inf),nStates),deflation.roots)
			rootsArray[i] = transpose(hcat(deflation.roots...))
	    end

		return rootsArray,pDeflations
	end
	@nograd findRoots

	""" differentiable deflation continuation method """
	function deflationContinuation( f::Function, u₀::Vector{Array{T,2}}, params::NamedTuple, paramlens::Lens,
		hyperparameters::ContinuationPar{T, S, E}, maxRoots::Int = 3, maxIter::Int=500; verbosity=0
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		maxIterContinuation,ds = hyperparameters.newtonOptions.maxIter,hyperparameters.ds
		J(u,p) = transpose(forward_jacobian(x->f(x,p),u)[end])

		rootsArray,pDeflations = findRoots( f, J, u₀, params, paramlens, hyperparameters, maxRoots, maxIter; verbosity=verbosity )
	    intervals = ([T(0.0),step(pDeflations)],[-step(pDeflations),T(0.0)])

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
	        			push!(branches,copy(branch)) end
	        		hyperparameters = @set hyperparameters.ds = -hyperparameters.ds
	            end
	    	end
	    end

		hyperparameters = @set hyperparameters.ds = ds
		return copy(branches), rootsArray
	end

	""" semi-supervised objective function """
	function loss( steady_states::CuBranch{U}, data::StateDensity{T},
		rates::Function, determinant::Function, curvature::Function,
		parameters::NamedTuple, hyperparameters::ContinuationPar;
		supervised::Bool=false, offset::Number=42 ) where {T<:Number,U<:Number}

		@unpack state,parameter = steady_states
		@unpack ds,pMin,pMax = hyperparameters

		N = length(parameter)
		Γ = 1.0/(pMax-pMin)

		# weighting towards valid solutions
		F = rates(state,parameter,parameters)
		solutions = exp.( -( typeof(F)<:Tuple ? .+(map(f->f.^2,F)...) : F.^2 ) / (2ds) ) / sqrt(4π*ds)

		if supervised # supervised signal

			bifurcation_density = solutions .* Γ./( Γ^2 .+ determinant(state,parameter,parameters).^2 )
			target_potential    =             -Γ./( Γ^2 .+ (data.bifurcations.-parameter').^2 )

			interaction = length(data.bifurcations) > 1 ? norm(bifurcation_density)^2 * ds : 0.0
			return sum(target_potential*bifurcation_density)*ds + interaction*Γ - offset

		else # unsupervised signal

			unsupervised = solutions .* curvature(state,parameter,parameters).^2
			return -log( sum(unsupervised)*ds )
		end
	end
end
