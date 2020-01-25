using Zygote: Buffer
using LinearAlgebra: UniformScaling,Dims,diagind,dot
import LinearAlgebra: Matrix

## Matrix construction from UniformScaling
function Matrix{T}(s::UniformScaling, dims::Dims{2}) where {T}
    A = Buffer([zero(T)],dims)
	A[:,:] = zeros(T,dims)
    v = T(s.λ)
    for i in diagind(dims...)
        @inbounds A[i] = v
    end
    return copy(A)
end
