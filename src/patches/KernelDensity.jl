using KernelDensity,Flux,Zygote,FillArrays
using Zygote: @adjoint, Buffer

import FFTW: fft,ifft,rfft,irfft
import Base: eps

################################################################### misc patches
@adjoint collect(x::Array) = collect(x), Δ -> (Δ,)
eps(t::Type{Complex{T}}) where {T<:Real} = eps(T)

import Zygote: accum
function accum(x::NamedTuple, y)
	if :len in keys(x)
		y
	else
		x
	end
end

function accum(y, x::NamedTuple)
	if :len in keys(x)
		y
	else
		x
	end
end

###################### derivatives of fft routines for automatic differentiation
fft(A::Fill, dims...) = fft(collect(A), dims...)
@adjoint function fft(A)
	return fft(A), Δ -> ( ifft(Δ)*length(A), )
end

################################################################################
ifft(A::Fill, dims...) = ifft(collect(A), dims...)
@adjoint function ifft(A)
	return ifft(A), Δ -> ( fft(Δ)/length(A), )
end

################################################################################
rfft(A::Fill, dims...) = rfft(collect(A), dims...)
@adjoint function rfft(A)
	return rfft(A), Δ -> ( irfft(Δ,length(A))*length(A), )
end

################################################################################
irfft(A::Fill, size::Integer, dims...) = irfft(collect(A), size, dims...)
@adjoint function irfft(A,size)
	return irfft(A,size), Δ -> ( rfft(Δ)/size, 0 )
end

# bypass mutations
using KernelDensity: RealVector,Weights
import KernelDensity: tabulate
function tabulate(data::RealVector, midpoints::R, weights::Weights=default_weights(data)) where R<:AbstractRange
    npoints = length(midpoints)
    s = step(midpoints)

    # Set up a grid for discretized data
	grid = Buffer(data,npoints); for i=1:npoints grid[i] = 0.0 end
    ainc = 1.0 / (sum(weights)*s*s)

    # weighted discretization (cf. Jones and Lotwick)
    for (i,x) in enumerate(data)
        k = searchsortedfirst(midpoints,x)
		j = k-1
        if 1 <= j <= npoints-1
            grid[j] += (midpoints[k]-x)*ainc*weights[i]
            grid[k] += (x-midpoints[j])*ainc*weights[i]
        end
    end

    # returns an un-convolved KDE
    UnivariateKDE(midpoints, copy(grid))
end

# function loss(θ)
# 	density = kde([1.0,θ],-3:0.1:3,bandwidth=0.1).density
# 	return sum(density)*step(-3:0.1:3)
# end
#
#
# θ = 4.0
# loss(θ)
#
# Zygote.gradient( loss, θ )
