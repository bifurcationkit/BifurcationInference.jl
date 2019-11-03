using KernelDensity
using Flux.Tracker: @grad, TrackedReal
using Flux

import FFTW: fft,ifft,rfft,irfft
import StatsBase: quantile

import Base: *,round
import Core.Integer

# derivatives of functions for automatic differentiation
fft(A::TrackedArray, dims...) = Tracker.track( fft, A, dims...)
fft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, dims...) where {T<:Real} = fft(Tracker.collect(A), dims...)

@grad function fft(A)
	return fft(Tracker.data(A)), Δ -> ifft(Δ)*length(A)
end

ifft(A::TrackedArray, dims...) = Tracker.track( ifft, A, dims...)
ifft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, dims...) where {T<:Real} = ifft(Tracker.collect(A), dims...)

@grad function ifft(A)
	return ifft(Tracker.data(A)), Δ -> fft(Δ)/length(A)
end

rfft(A::TrackedArray, dims...) = Tracker.track( rfft, A, dims...)
rfft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, dims...) where {T<:Real} = rfft(Tracker.collect(A), dims...)

@grad function rfft(A)
	return rfft(Tracker.data(A)), Δ -> irfft(Δ)*length(A)
end

irfft(A::TrackedArray, size::Integer, dims...) = Tracker.track( irfft, A, size, dims...)
irfft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, size::Integer, dims...) where {T<:Real} = irfft(Tracker.collect(A), size, dims...)

@grad function irfft(A,size)
	return irfft(Tracker.data(A),size), Δ -> rfft(Δ)/size
end

# misc patches
quantile(x::TrackedArray, p::AbstractArray) = quantile(x.data, p)
round(::Type{R}, t::TrackedReal{T}) where {R<:Real,T<:Real} = round(R, t.data)
# round(t::TrackedReal, mode::RoundingMode) = round(t.data, mode)
# Integer(x::TrackedReal) = Integer(x.data)

Tracker.Tracked
kde( param(randn(100)), -3:0.1:3, bandwidth=0.1 )

z = randn(100).+1im
x = randn(100)
eltype(z)<:Complex
param(z)[1]
typeof(param(x)[1])

x[1] *= cf(Normal(),1.0)
x[1] *= cf(Normal(),param(1.0))

import Base: *

*(a::Tracker.Tracked, b::Complex) = Tracker.track(*, a, b)
@grad function *(a,b)
	return Tracker.data(a)*Tracker.data(b), Δ -> fft(Δ)/length(A)
end

cf(Normal(),1.0)
x * cf(Normal(),1.0)
x

Tracker.Dual(1+1.0im,1)

track(*, a, b)
