module FluxContinuation

	using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig, AbstractLinearSolver, AbstractEigenSolver, DeflationOperator
	using PseudoArcLengthContinuation; const Cont = PseudoArcLengthContinuation

	include("patches/PseudoArcLengthContinuation.jl")
	include("patches/LinearAlgebra.jl")
	include("patches/Flux.jl")
	export continuation

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
	function deflationContinuation( f::Function, J::Function, u₀::Vector{T}, parameters::ContinuationPar{T, S, E},
		printsolution::Function = u->u[1], finaliseSolution::Function = (_,_,_,_) -> true, maxBranches::Int = 10, nDeflations=2
			) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		branchParameters = deepcopy(parameters)
		pDeflations = range(branchParameters.pMin,branchParameters.pMax,length=nDeflations)
		branches,pStep = [],step(pDeflations)

		branch, = Cont.continuation( f, J, u₀, branchParameters.pMin, branchParameters)
		branches = Buffer([branch],2*nDeflations*maxBranches); for i=1:2*nDeflations*maxBranches branches[i] = branch end
		n = 0

		for i=1:nDeflations
			roots = Buffer([Inf],maxBranches,length(u₀)); for i=1:maxBranches roots[i,:] = fill(Inf,length(u₀)) end
			deflation = DeflationOperator(1.0, dot, 1.0, roots, 0)
			u = copy(u₀)

			while length(deflation) < maxBranches # search for roots
				u, _, converged, _ = Cont.newtonDeflated( u->f(u,pDeflations[i]), u->J(u,pDeflations[i]),
					u.+5*abs(branchParameters.ds), branchParameters.newtonOptions, deflation)
				if converged push!(deflation,u) else break end
			end

			# continue from roots
			deflation.roots = copy(deflation.roots)
			for j=1:deflation.n_roots
				u = deflation.roots[j,:]

				# forwards branch
				branchParameters.pMin,branchParameters.pMax = pDeflations[i], pDeflations[i] + pStep
				forward, = Cont.continuation( f, J, u, branchParameters.pMin+branchParameters.newtonOptions.tol,
					branchParameters; printsolution = printsolution, finaliseSolution = finaliseSolution)

				if branchParameters.pMax <= maximum(pDeflations)
					n +=1; branches[n] = forward
				end
				branchParameters.ds *= -1.0

				# backwards branch
				branchParameters.pMin,branchParameters.pMax = pDeflations[i]-pStep, pDeflations[i]
				backward, = Cont.continuation( f, J, u, branchParameters.pMax-branchParameters.newtonOptions.tol,
					branchParameters; printsolution = printsolution, finaliseSolution = finaliseSolution)

				if branchParameters.pMin >= minimum(pDeflations)
					n +=1; branches[n] = backward
				end
				branchParameters.ds *= -1.0
			end
		end
		return branches[map( branch -> branch.n_points > 1, copy(branches) )]
	end
end
