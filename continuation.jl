using PseudoArcLengthContinuation, Flux
using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig
const Cont = PseudoArcLengthContinuation

import Base: copyto!
copyto!(x::TrackedArray, y::TrackedArray) where T = x = copy(y)

import PseudoArcLengthContinuation: Newton, minus!
minus!(x::TrackedArray, y::TrackedArray) = x = x .- y
minus!(x::TrackedArray, y) = x = x .- y

function fdTracked(F, x::AbstractVector; δ = 1e-9)
	f = F(x)
	N = length(x)
	J = zeros(eltype(f), N, N)

	Δx = zeros(N)
	for i=1:N
		Δx[i] += δ
		J[:, i] .= (F(x.+Δx) .- F(x)) / δ
		Δx[i] -= δ
	end
	return J
end

unpack(curve::ContResult) = Tracker.collect(curve.branch[1,:]),Tracker.collect(curve.branch[2,:])

function continuation( f, u₀,p₀ ; kwargs...)
	T = eltype(u₀)

	optnew = Cont.NewtonPar{T,typeof(DefaultLS()), typeof(DefaultEig())}(verbose = false)

	u₀, _, stable = Cont.newton( u -> f(u,p₀), u0 -> fdTracked(u -> f(u,p₀), u0), u₀, optnew )

	branch, _, _ = Cont.continuation( f, (u0, p) -> fdTracked(u -> f(u, p), u0), u₀,p₀,

		ContinuationPar{T,typeof(DefaultLS()), typeof(DefaultEig())}(
			pMin=kwargs[:pMin],pMax=kwargs[:pMax],ds=kwargs[:ds],
			maxSteps=kwargs[:maxSteps])
	)
	return Tracker.data.(u₀), unpack(branch)...
end
