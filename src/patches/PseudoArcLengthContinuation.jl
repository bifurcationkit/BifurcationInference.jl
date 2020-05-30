import PseudoArcLengthContinuation: copy,rmul!,minus!,axpy!
using LinearAlgebra: BlasFloat

##################### override in-place operations for BorderedArray
copy(b::BorderedArray) = BorderedArray(copy(b.u),copy(b.p))

rmul!(x::T, y::U) where {T <: AbstractArray, U <: Number} = (x = x .* y; x)
function rmul!(A::BorderedArray{vectype, Tv}, a::T, b::T) where {vectype, Tv <: Number, T <: Number }
	A.u = rmul!(A.u,a)
	A.p = A.p * b
	return A
end

minus!(x::T, y::T) where {T <: AbstractArray} = (x = x .- y; x)
function minus!(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Number}
	x.u = minus!(x.u,y.u)
	x.p = x.p - y.p
	return x
end

axpy!(a::Number, x::Union{DenseArray{T},StridedVector{T}}, y::Union{DenseArray{T},StridedVector{T}}) where T<:BlasFloat = ( y = y .+ a .* x ; y)
function axpy!(a::T, X::BorderedArray{Tv1, Tp1}, Y::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, T <: Number, Tp1 <: Number, Tp2 <: Number}
	Y.u = axpy!(a, X.u, Y.u )
	Y.p = a * X.p + Y.p
	return Y
end
