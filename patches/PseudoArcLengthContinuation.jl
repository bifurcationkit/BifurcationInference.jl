using Flux,PseudoArcLengthContinuation
using Flux.Tracker: TrackedReal
using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig
const Cont = PseudoArcLengthContinuation

import Base: vcat,round
import Plots: plot, plot!
import LinearAlgebra: eigen,eigen!

# methods for unpacking ContResult
unpack(curve::ContResult) = tuple([ curve.branch[i,:] for i=1:size(curve.branch)[1]-2 ]...)
initial_state(curve::ContResult) = [ param(curve.branch[i,1].data) for i=2:size(curve.branch)[1]-2 ]

# patches for Cont.continuation()
vcat(p::TrackedReal{T}, u::Vector{TrackedReal{T}}, i::Int64, ds::TrackedReal{T}) where {T<:Real} = [ p, u..., i, ds ]
round(x::TrackedReal{T}, mode::RoundingMode; kwargs...) where {T<:Real} = round(x.data, mode; kwargs...)

eigen(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen( Tracker.data.(x); kwargs...)
eigen!(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen!( Tracker.data.(x); kwargs...)


""" extension of co-dimension one parameter continuation methods to Array{TrackedReal} type """
function continuation( f::Function, J::Function, u₀::Vector{T}, p₀::T,
	printsolution = u->u ; kwargs...) where {T<:Number}

	options = Cont.NewtonPar{T, typeof(DefaultLS()), typeof(DefaultEig())}(verbose=false,maxIter=kwargs[:maxIter])
	return Cont.continuation( f,J, u₀,p₀,
		Cont.ContinuationPar{T,typeof(DefaultLS()), typeof(DefaultEig())}(

			pMin=kwargs[:pMin],pMax=kwargs[:pMax],ds=kwargs[:ds],
			maxSteps=kwargs[:maxSteps], newtonOptions = options,

			computeEigenValues = kwargs[:computeEigenValues]);
			printsolution = printsolution )

end

# patches for Cont.plotBranch()
plot(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x); kwargs...)
plot!(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x); kwargs...)

plot(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x), Tracker.data.(y); kwargs...)
plot!(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), Tracker.data.(y); kwargs...)

plot(x::AbstractArray,y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( x, Tracker.data.(y); kwargs...)
plot!(x::AbstractArray,y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( x, Tracker.data.(y); kwargs...)

plot(x::Vector{TrackedReal{T}},y::AbstractArray; kwargs...) where {T<:Real} = plot( Tracker.data.(x), y; kwargs...)
plot!(x::Vector{TrackedReal{T}},y::AbstractArray; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), y; kwargs...)