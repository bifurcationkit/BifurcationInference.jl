module FluxContinuation

	using Flux, Plots, LinearAlgebra, Printf
	include("patches/PseudoArcLengthContinuation.jl")
	export continuation

	""" extension of co-dimension one parameter continuation methods to Array{TrackedReal} type """
	function continuation( f::Function, J::Function, u₀::Vector{T}, p₀::T,
		printsolution::Function = u->u[1]; kwargs...) where {T<:Number}
	
		function updateInitial( state::BorderedArray, gradient::BorderedArray, step::Int64, contResult::ContResult )
			if step == 0 u₀ = param.(Tracker.data.(state.u)) end
			return true
		end
	
		options = Cont.NewtonPar{T, typeof(DefaultLS()), typeof(DefaultEig())}(verbose=false,maxIter=kwargs[:maxIter])
		bifurcations, = Cont.continuation( f,J, u₀,p₀,
			Cont.ContinuationPar{T,typeof(DefaultLS()), typeof(DefaultEig())}(
	
				pMin=kwargs[:pMin],pMax=kwargs[:pMax],ds=kwargs[:ds],
				maxSteps=kwargs[:maxSteps], newtonOptions = options,
	
				computeEigenValues = kwargs[:computeEigenValues]);
				printsolution = printsolution, finaliseSolution = updateInitial )
	
		return bifurcations, u₀
	
	end
end