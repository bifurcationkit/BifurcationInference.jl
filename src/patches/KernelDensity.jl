using KernelDensity,Flux,Zygote,FillArrays
using Tracker: @grad, TrackedReal,TrackedComplex
using Zygote: @adjoint

import FFTW: fft,ifft,rfft,irfft
import StatsBase: quantile

import Base: *,round,eps
import Core.Integer

################################################################### misc patches
@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)
quantile(x::TrackedArray, p::AbstractArray) = quantile(x.data, p)
round(::Type{R}, t::Union{TrackedReal,TrackedComplex}) where R<:Real = round(R, t.data)
eps(t::Type{Complex{T}}) where {T<:Real} = eps(T)

###################### derivatives of fft routines for automatic differentiation
fft(A::TrackedArray, dims...) = Tracker.track( fft, A, dims...)
fft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, dims...) where {T<:Real} = fft(Tracker.collect(A), dims...)
fft(A::Fill, dims...) = fft(collect(A), dims...)

@grad function fft(A)
	return fft(Tracker.data(A)), Δ -> ( ifft(Δ)*length(A), )
end

@adjoint function fft(A)
	return fft(A), Δ -> ( ifft(Δ)*length(A), )
end

################################################################################
ifft(A::TrackedArray, dims...) = Tracker.track( ifft, A, dims...)
ifft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, dims...) where {T<:Real} = ifft(Tracker.collect(A), dims...)
ifft(A::Fill, dims...) = ifft(collect(A), dims...)

@grad function ifft(A)
	return ifft(Tracker.data(A)), Δ -> ( fft(Δ)/length(A), )
end

@adjoint function ifft(A)
	return ifft(A), Δ -> ( fft(Δ)/length(A), )
end

################################################################################
rfft(A::TrackedArray, dims...) = Tracker.track( rfft, A, dims...)
rfft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, dims...) where {T<:Real} = rfft(Tracker.collect(A), dims...)
rfft(A::Fill, dims...) = rfft(collect(A), dims...)

@grad function rfft(A)
	return rfft(Tracker.data(A)), Δ -> ( irfft(Δ,length(A))*length(A), )
end

@adjoint function rfft(A)
	return rfft(A), Δ -> ( irfft(Δ,length(A))*length(A), )
end

################################################################################
irfft(A::TrackedArray, size::Integer, dims...) = Tracker.track( irfft, A, size, dims...)
irfft(A::Union{Array{TrackedReal{T}},Array{Complex{TrackedReal{T}}}}, size::Integer, dims...) where {T<:Real} = irfft(Tracker.collect(A), size, dims...)
irfft(A::Fill, size::Integer, dims...) = irfft(collect(A), size, dims...)

@grad function irfft(A,size)
	return irfft(Tracker.data(A),size), Δ -> ( rfft(Δ)/size, 0 )
end

@adjoint function irfft(A,size)
	return irfft(A,size), Δ -> ( rfft(Δ)/size, 0 )
end


# function lossA(θ)
# 	data = θ.*collect(-3:0.1:3)
# 	density = kde(data,-3:0.1:3,bandwidth=0.1).density
# 	return sum(density)
# end
#
# function lossB(θ)
# 	data = θ.*collect(-3:0.1:3)
# 	density = irfft(rfft(data).*exp.(im*pi*(1:length(data)>>1+1)),length(data))
# 	return sum(density)
# end

# θ = 2.0
# lossA(θ),lossB(θ)
#
# Tracker.gradient( lossA, θ )
# Zygote.gradient( lossA, θ )
#
# Tracker.gradient( lossB, θ )
# Zygote.gradient( lossB, θ )
