using PseudoArcLengthContinuation, Flux
using PseudoArcLengthContinuation: ContResult,AbstractLinearSolver,EigenSolver
const Cont = PseudoArcLengthContinuation

unpack(curve::ContResult) = Tracker.collect(curve.branch[1,:]),Tracker.collect(curve.branch[2,:])
function continuation( f, u₀,p₀ ; kwargs...)

	u₀, _, stable = Cont.newton( u -> f(u,p₀), u₀, Cont.NewtonPar(verbose=false) )
	branch, _, _ = Cont.continuation( f, u₀,p₀,

		ContinuationPar{eltype(u₀),AbstractLinearSolver,EigenSolver}(
			pMin=kwargs[:pMin],pMax=kwargs[:pMax],ds=kwargs[:ds],
			maxSteps=kwargs[:maxSteps])
	)
	return Tracker.data.(u₀), unpack(branch)...
end
