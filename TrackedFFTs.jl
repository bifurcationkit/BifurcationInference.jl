using AbstractFFTs: Plan
using Flux.Tracker: TrackedReal

import AbstractFFTs: plan_fft, plan_rfft, plan_bfft, plan_brfft, plan_inv
import AbstractFFTs: realfloat, complexfloat
import Base: *, mul!


mutable struct TrackedPlan{T} <: Plan{T}
    region; pinv::Plan{T}
    TrackedPlan{T}(region) where {T} = new{T}(region)
end

mutable struct InverseTrackedPlan{T} <: Plan{T}
    region; pinv::Plan{T}
    InverseTrackedPlan{T}(region) where {T} = new{T}(region)
end


complexfloat(x::Array{TrackedReal{T}}) where {T<:Real} = x .+ 0im
realfloat(x::Array{TrackedReal{T}}) where {T<:Real} = x
plan_inv(p::TrackedPlan{T}) where {T} = InverseTrackedPlan{T}

plan_fft(x::Array{Complex{TrackedReal{T}}}, region; kwargs...) where {T<:Real} = TrackedPlan{Array{Complex{TrackedReal{T}}}}(region)
plan_bfft(x::Array{Complex{TrackedReal{T}}}, region; kwargs...) where {T<:Real} = InverseTrackedPlan{Array{Complex{TrackedReal{T}}}}(region)

plan_rfft(x::Array{TrackedReal{T}}, region; kwargs...) where {T<:Real} = TrackedPlan{Array{TrackedReal{T}}}(region)
plan_brfft(x::Array{Complex{TrackedReal{T}}}, N::Int64, region; kwargs...) where {T<:Real} = InverseTrackedPlan{Array{TrackedReal{T}}}(region)

*(p::TrackedPlan, x::AbstractArray) = mul!(copy(x), p, x)
*(p::InverseTrackedPlan, x::AbstractArray) = mul!(copy(x), p, x)

mul!(y::Array{TrackedReal{T}}, p::TrackedPlan, x::Array{TrackedReal{T}}) where {T<:Real} = rdft!(y, x, -1)
mul!(y::Array{TrackedReal{T}}, p::InverseTrackedPlan, x::Array{TrackedReal{T}}) where {T<:Real} = rdft!(y, x, 1)

mul!(y::Array{Complex{TrackedReal{T}}}, p::TrackedPlan, x::Array{Complex{TrackedReal{T}}}) where {T<:Real} = dft!(y, x, -1)
mul!(y::Array{Complex{TrackedReal{T}}}, p::InverseTrackedPlan, x::Array{Complex{TrackedReal{T}}}) where {T<:Real} = dft!(y, x, 1)


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

function rdft!(y::AbstractArray, x::AbstractArray, sign::Int)
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    fill!(y, zero(complex(float(eltype(x)))))
    c = sign * 2π / n
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(c*j*k)
    end
    return y
end





# unit tests
using FFTW: fft,rfft,ifft,irfft
using Flux

input,output = rand(1000),rand(501)
N = length(input)

@assert all( fft(input) .≈ fft(param.(input)) )
@assert all( ifft(input) .≈ ifft(param.(input)) )

@assert all( rfft(input) .≈ rfft(param.(input)) )
@assert all( irfft(output,N) .≈ irfft(param.(output),N) )
