using Flux,PseudoArcLengthContinuation
using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig
const Cont = PseudoArcLengthContinuation

import Base.vcat
import Plots: plot, plot!
import PseudoArcLengthContinuation: eigen!
using Flux.Tracker: TrackedReal,TrackedArray

unpack(curve::ContResult) = tuple([ curve.branch[i,:] for i=1:size(curve.branch)[1]-2 ]...)
initial_state(curve::ContResult) = [ param(curve.branch[i,1].data) for i=2:size(curve.branch)[1]-2 ]

plot(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x); kwargs...)
plot!(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x); kwargs...)

plot(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x), Tracker.data.(y); kwargs...)
plot!(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), Tracker.data.(y); kwargs...)
plot!(x::AbstractArray,y::TrackedArray; kwargs...) = plot!( x, y.data; kwargs...)

vcat(p::TrackedReal{T}, u::Vector{TrackedReal{T}}, i::Int64, ds::TrackedReal{T}) where {T<:Real} = [ p, u..., i, ds ]
eigen!(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen!( Tracker.data.(x); kwargs...)

function continuation( f::Function, J::Function,
	u₀::Vector{TrackedReal{T}}, p₀::TrackedReal{T},
	printsolution = u->u ; kwargs...) where {T<:Real}

	options = Cont.NewtonPar{TrackedReal{T}, typeof(DefaultLS()), typeof(DefaultEig())}(verbose=false,maxIter=kwargs[:maxIter])
	return Cont.continuation( f,J, u₀,p₀,
		ContinuationPar{TrackedReal{T},typeof(DefaultLS()), typeof(DefaultEig())}(

			pMin=kwargs[:pMin],pMax=kwargs[:pMax],ds=kwargs[:ds],
			maxSteps=kwargs[:maxSteps], newtonOptions = options,

			computeEigenValues = kwargs[:computeEigenValues]);
			printsolution = printsolution )
end
