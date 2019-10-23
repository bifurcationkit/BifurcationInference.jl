using Flux,PseudoArcLengthContinuation
using Flux.Tracker: TrackedReal
using PseudoArcLengthContinuation: ContResult, DefaultLS, DefaultEig
const Cont = PseudoArcLengthContinuation

import Base: vcat,round
import Plots: plot, plot!
import LinearAlgebra: eigen,eigen!

# methods for unpacking ContResult
unpack(curve::ContResult) = tuple([ curve.branch[i,:] for i=1:size(curve.branch)[1]-2 ]...)

# patches for Cont.continuation()
vcat(p::TrackedReal{T}, u::Vector{TrackedReal{T}}, i::Int64, ds::TrackedReal{T}) where {T<:Real} = [ p, u..., i, ds ]
vcat(p::TrackedReal{T}, u::TrackedReal{T}, i::Int64, ds::TrackedReal{T}) where {T<:Real} = [ p, u, i, ds ]
round(x::TrackedReal{T}, mode::RoundingMode; kwargs...) where {T<:Real} = round(x.data, mode; kwargs...)

eigen(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen( Tracker.data.(x); kwargs...)
eigen!(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen!( Tracker.data.(x); kwargs...)


""" extension of co-dimension one parameter continuation methods to Array{TrackedReal} type """
function continuation( f::Function, J::Function, u₀::Vector{T}, p₀::T,
	printsolution = u->u[1]; kwargs...) where {T<:Number}

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

# patches for Cont.plotBranch()
plot(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x); kwargs...)
plot!(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x); kwargs...)

plot(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x), Tracker.data.(y); kwargs...)
plot!(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), Tracker.data.(y); kwargs...)

plot(x::AbstractArray,y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( x, Tracker.data.(y); kwargs...)
plot!(x::AbstractArray,y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( x, Tracker.data.(y); kwargs...)

plot(x::Vector{TrackedReal{T}},y::AbstractArray; kwargs...) where {T<:Real} = plot( Tracker.data.(x), y; kwargs...)
plot!(x::Vector{TrackedReal{T}},y::AbstractArray; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), y; kwargs...)
