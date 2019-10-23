using Flux.Tracker: TrackedReal
using Flux

import Base: vcat,round
import Plots: plot, plot!
import LinearAlgebra: eigen,eigen!

# patches for Cont.continuation()
vcat(p::TrackedReal{T}, u::Vector{TrackedReal{T}}, i::Int64, ds::TrackedReal{T}) where {T<:Real} = [ p, u..., i, ds ]
vcat(p::TrackedReal{T}, u::TrackedReal{T}, i::Int64, ds::TrackedReal{T}) where {T<:Real} = [ p, u, i, ds ]
round(x::TrackedReal{T}, mode::RoundingMode; kwargs...) where {T<:Real} = round(x.data, mode; kwargs...)

eigen(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen( Tracker.data.(x); kwargs...)
eigen!(x::Array{TrackedReal{T}}; kwargs...) where {T<:Real} = eigen!( Tracker.data.(x); kwargs...)

# patches for Cont.plotBranch()
plot(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x); kwargs...)
plot!(x::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x); kwargs...)

plot(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( Tracker.data.(x), Tracker.data.(y); kwargs...)
plot!(x::Vector{TrackedReal{T}},y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), Tracker.data.(y); kwargs...)

plot(x::AbstractArray,y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot( x, Tracker.data.(y); kwargs...)
plot!(x::AbstractArray,y::Vector{TrackedReal{T}}; kwargs...) where {T<:Real} = plot!( x, Tracker.data.(y); kwargs...)

plot(x::Vector{TrackedReal{T}},y::AbstractArray; kwargs...) where {T<:Real} = plot( Tracker.data.(x), y; kwargs...)
plot!(x::Vector{TrackedReal{T}},y::AbstractArray; kwargs...) where {T<:Real} = plot!( Tracker.data.(x), y; kwargs...)
