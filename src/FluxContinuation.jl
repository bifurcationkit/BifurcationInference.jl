module FluxContinuation

	using PseudoArcLengthContinuation: AbstractLinearSolver, AbstractEigenSolver, DeflationOperator, PALCIterable
	using PseudoArcLengthContinuation; const Cont = PseudoArcLengthContinuation
	using Zygote: Buffer, @nograd

	using Setfield: @set,setproperties
	using Dates: now
	using Logging

	using LinearAlgebra: dot
	using StatsBase: norm

	using Plots.PlotMeasures
	using LaTeXStrings
	using Plots

	include("patches/Flux.jl")
	include("Structures.jl")
	include("Utils.jl")

	export StateDensity,Branch,deflationContinuation,findRoots
	export getParameters,updateParameters,loss

	@nograd now,string

	""" root finding with newton deflation method"""
	function findRoots( f::Function, J::Function, u₀::Vector{Array{T,2}}, parameters::ContinuationPar{T, S, E},
		maxRoots::Int = 3, maxIter::Int=500 ) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		pDeflations = range(parameters.pMin,parameters.pMax,length=length(u₀))
		rootsArray = similar(u₀)
		_,nStates = size(u₀[1])

		parameters = @set parameters.newtonOptions.maxIter = maxIter
		with_logger(NullLogger()) do # silence newton convergence errors
		    for (i,us) in enumerate(u₀)

				deflation = DeflationOperator(1.0, dot, 1.0, [fill(Inf,nStates)] )
				converged = true

		        for u in eachrow(us) # update existing roots
		    		u, _, converged, _ = Cont.newton( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
			    		u.+parameters.ds, parameters.newtonOptions, deflation)
		    		if converged push!(deflation,u) else break end
		        end

				if converged # search for new roots
					u = fill(0.0,nStates)
					while length(deflation) < maxRoots

						u, _, converged, _ = Cont.newton( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
							u.+parameters.ds, parameters.newtonOptions, deflation)

						if converged & all(map( v -> !isapprox(u,v,atol=2*parameters.ds), deflation.roots))
							push!(deflation,u)
						else break end
					end
				end

				filter!(root->root≠fill(Inf,nStates),deflation.roots)
				rootsArray[i] = transpose(hcat(deflation.roots...))
		    end
		end

		return rootsArray,pDeflations
	end
	@nograd findRoots

	""" differentiable deflation continuation method """
	function deflationContinuation( f::Function, J::Function, u₀::Vector{Array{T,2}}, parameters::ContinuationPar{T, S, E},
		maxRoots::Int = 3, maxIter::Int=500 ) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		maxIterContinuation,ds = parameters.newtonOptions.maxIter,parameters.ds
		rootsArray,pDeflations = findRoots( f, J, u₀, parameters, maxRoots, maxIter )

	    nRoots = map( us->( (nRoots,nStates)=size(us); nRoots), rootsArray)
	    intervals = ([0.0,step(pDeflations)],[-step(pDeflations),0.0])

		# continuation per root
		branches = Buffer([ Branch(T) ], 2*sum(nRoots)-nRoots[end]-nRoots[1] ); n = 0
		parameters = @set parameters.newtonOptions.maxIter = maxIterContinuation

	    for (i,us) in enumerate(rootsArray)
	        for u in eachrow(us)

	            # forwards and backwards branches
	            for (pMin,pMax) in intervals

					parameters = setproperties(parameters;
						pMin=pDeflations[i]+pMin, pMax=pDeflations[i]+pMax,
						ds=sign(parameters.ds)*ds)

	                # main continuation method
					branch = Branch(T)
					iterator = PALCIterable( f, J, u, pDeflations[i]+parameters.ds, parameters)

					for state in iterator

						push!(branch.state,copy(getx(state)))
						push!(branch.parameter,getp(state))

						Cont.computeEigenvalues!(iterator, state)
						push!(branch.eigvals,state.eigvals)

						if Cont.detectBifucation(state)
							push!(branch.bifurcations, (state=copy(getx(state)),parameter=getp(state)) )
						end
					end

	        		if parameters.pMax <= maximum(pDeflations) && parameters.pMin >= minimum(pDeflations)
	        			n +=1; branches[n] = branch end
	        		parameters = @set parameters.ds = -parameters.ds
	            end
	    	end
	    end

		parameters = @set parameters.ds = ds
		return copy(branches), rootsArray
	end

	""" semi-supervised objective function """
	function loss(steady_states::Vector{Branch{T}}, data::StateDensity{T}, curvature::Function) where {T<:Number}

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
end
