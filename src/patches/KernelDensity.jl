using AbstractFFTs: Plan
using Flux.Tracker: TrackedReal
using Flux

import AbstractFFTs: plan_fft, plan_rfft, plan_bfft, plan_brfft, plan_inv
import AbstractFFTs: _fftfloat,fftfloat,complexfloat,realfloat

import Base: *,round
import Core.Integer

mutable struct TrackedPlan{T} <: Plan{T}
    region; pinv::Plan{T}; N::Int64; real::Bool
    TrackedPlan{T}(region) where {T} = new{T}(region)
end
mutable struct InverseTrackedPlan{T} <: Plan{T}
    region; pinv::Plan{T}; N::Int64; real::Bool
    InverseTrackedPlan{T}(region) where {T} = new{T}(region)
end
plan_inv(p::TrackedPlan{T}) where {T} = InverseTrackedPlan{T}

# for compatibility with Array{TrackedReal} inputs
_fftfloat(::Type{TrackedReal{T}}) where {T<:Real} = TrackedReal{T}
_fftfloat(::Type{Complex{TrackedReal{T}}}) where {T<:Real} = Complex{TrackedReal{T}}

# for compatibility with TrackedArray inputs
complexfloat(x::TrackedArray) = copy(x)
realfloat(x::TrackedArray) = copy(x)

function copy(x::TrackedArray)
    y = Array{eltype(x)}(undef, map(length, axes(x)))
    circcopy!(y, x)
end

# abstract routine extensions
function plan_fft(x::Union{Array{Complex{TrackedReal{T}}},TrackedArray{Array{Complex{T}}}}, region; kwargs...) where {T<:Real}
    plan = TrackedPlan{Array{Complex{TrackedReal{T}}}}(region)
    plan.N = length(x); plan.real = false
    return plan
end

function plan_bfft(x::Union{Array{Complex{TrackedReal{T}}},TrackedArray{Array{Complex{T}}}}, region; kwargs...) where {T<:Real}
    plan = InverseTrackedPlan{Array{Complex{TrackedReal{T}}}}(region)
    plan.N = length(x); plan.real = false
    return plan
end

function plan_rfft(x::Union{Array{TrackedReal{T}},TrackedArray{Array{T}}}, region; kwargs...) where {T<:Real}
    plan = TrackedPlan{Array{TrackedReal{T}}}(region)
    plan.N = length(x)>>1+1; plan.real = true
    return plan
end

function plan_brfft(x::Union{Array{Complex{TrackedReal{T}}},TrackedArray{Array{T}}}, N::Int64, region; kwargs...) where {T<:Real}
    plan = InverseTrackedPlan{Array{Complex{TrackedReal{T}}}}(region)
    plan.N = N; plan.real = true
    return plan
end

*(p::TrackedPlan, x::AbstractArray) where {T<:Real} = mul!( zeros(eltype(x),p.N) .+ 0im, p, x)
*(p::InverseTrackedPlan, x::AbstractArray) where {T<:Real} = mul!( zeros(eltype(x),p.N) .+ 0im, p, x)

function mul!(y::AbstractArray, p::TrackedPlan, x::AbstractArray) where {T<:Real}
    if p.real rdft!(y,x) else dft!(y,x,-1) end
end

function mul!(y::AbstractArray, p::InverseTrackedPlan, x::AbstractArray) where {T<:Real}
    if p.real irdft!( real(y),x,p.N) else dft!(y,x,1) end
end

round(::Type{R}, t::TrackedReal{T}) where {R<:Real,T<:Real} = round(R, t.data)
round(t::TrackedReal, mode::RoundingMode) = round(t.data, mode)
Integer(x::TrackedReal) = Integer(x.data)

############################################################# fft algorithms
# TODO inefficient N^2.. I want to use an NlogN algorithm

function dft!(y::AbstractArray, x::AbstractArray, sign::Int)
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    fill!(y, zero(complex(float(eltype(x)))))
    c = sign * 2π / n
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(c*j*k)
    end
    return y
end

function rdft!(y::AbstractArray, x::AbstractArray)
    n = length(x)
    length(y) == n>>1+1 || throw(DimensionMismatch())
    fill!(y, zero(complex(float(eltype(x)))))
    c = - 2π / n
    @inbounds for j = 0:n-1, k = 0:n>>1
        y[k+1] += x[j+1] * cis(c*j*k)
    end
    return y
end

# TODO unit test accuracy fails for N even
function irdft!(y::AbstractArray, x::AbstractArray, N::Int64)
    n = length(x)
    length(y) == N || throw(DimensionMismatch())
    fill!(y, zero(complex(float(eltype(x)))))
    c = 2π / N
    @inbounds for j = 0:n-1, k = 0:N-1
        y[k+1] += 2*real(x[j+1]*cis(c*j*k)) - x[1]/n
    end
    return y
end
