import PseudoArcLengthContinuation: copy,rmul!,minus!,axpy!

copy(b::BorderedArray) = BorderedArray(copy(b.u), copy(b.p))
function rmul!(A::BorderedArray{vectype, Tv}, a::T, b::T) where {vectype, Tv <: Number, T <: Number }
	A.u = A.u * a
	A.p = A.p * b
	return A
end

function minus!(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Number}
	x.u = x.u - y.u
	x.p = x.p - y.p
	return x
end

function axpy!(a::T, X::BorderedArray{Tv1, Tp1}, Y::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, T <: Number, Tp1 <: Number, Tp2 <: Number}
	Y.u = a * X.u + Y.u
	Y.p = a * X.p + Y.p
	return Y
end
