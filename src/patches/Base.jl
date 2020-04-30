import Base: unique

function unique(A::AbstractVector, tol::Float64)
	uniqueVector = Vector{eltype(A)}()
	for a in A
		if !any(isapprox.(a,uniqueVector,atol=tol))
			push!(uniqueVector,a)
		end
	end
	return uniqueVector
end
