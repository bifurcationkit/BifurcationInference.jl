module FluxContinuation

	using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig, AbstractLinearSolver, AbstractEigenSolver
	using PseudoArcLengthContinuation
	const Cont = PseudoArcLengthContinuation

	include("patches/PseudoArcLengthContinuation.jl")
	export continuation

	""" extension of co-dimension one parameter continuation methods work with Zygote """
	function continuation( f::Function, J::Function, u₀::Vector{T}, p₀::T, parameters::ContinuationPar{T, S, E},
		printsolution::Function = u->u[1], finaliseSolution::Function = (_,_,_,_) -> true
		) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

		function update( state::BorderedArray, gradient::BorderedArray, step::Int64, contResult::ContResult )
			if step == 0 u₀ = state.u end
			return finaliseSolution(state,gradient,step,contResult)
		end

		bifurcations, = Cont.continuation( f,J, u₀,p₀, parameters;
			printsolution = printsolution, finaliseSolution = update )

		return bifurcations, u₀

	end
end
