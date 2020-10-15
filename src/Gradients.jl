################################################################################# loss gradient
function ∇loss( F::Ref{<:Function}, branches::AbstractVector{<:Branch}, θ::Ref{<:AbstractVector}, targets::Ref{<:AbstractVector}; kwargs...)

	∇L = sum( ∇marginal_likelihood.(F,branches,θ,targets; kwargs...) )
	L = sum( marginal_likelihood.(F,branches,θ,targets; kwargs...) )

	∇Z = sum( ∇normalisation.(F,branches,θ) )
	Z = sum( normalisation.(F,branches,θ) )

	return log(Z)-log(L), ∇Z/Z - ∇L/L
end

########################################################################### jacobians

# statespace jacobian
function ∂Fu(F::Function,z::BorderedArray,θ::AbstractVector)
	return jacobian( u -> F(   u, (θ=θ,p=z.p) ), z.u )
end

# parameter jacobian
function ∂Fθ(F::Function,z::BorderedArray,θ::AbstractVector)
	return jacobian( θ -> F( z.u, (θ=θ,p=z.p) ), θ )
end

# augmented jacobian
function ∂Fz(F::Function,z::BorderedArray,θ::AbstractVector)
	return jacobian( z -> F( z[Not(end)], (θ=θ,p=z[end]) ), [z.u; z.p] )
end

################################### gradient terms due to changing integration region dz
function ∇region( F::Function, integrand::Function, z::BorderedArray, θ::AbstractVector)

	∂z = jacobian( z -> velocity(F,integrand,z,θ), [z.u; z.p] )
	idx = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]

	# div(z) = tr(∂z) for each component θ
	return [ tr(∂z[i,:]) for i ∈ idx ]
end

function velocity( F::Function, integrand::Function, z::AbstractVector, θ::AbstractVector)
	z = BorderedArray(z[Not(end)],z[end])
	return -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * integrand(F,z,θ)
end

################################################################################
function ∇likelihood( F::Function, z::BorderedArray, θ::AbstractVector, targets::AbstractVector; kwargs...)
	integrand(F,z,θ) = likelihood(F,z,θ,targets; kwargs...)
	return gradient(θ->integrand(F,z,θ),θ) .+ ∇region(F,integrand,z,θ)
end

function ∇marginal_likelihood( F::Function, branch::Branch, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return branch.ds'∇likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); kwargs...)
end

function ∇normalisation(F::Function,branch::Branch,θ::AbstractVector)
	return gradient( θ -> normalisation(F,branch,θ), θ ) .+ branch.ds'∇region.( Ref(F), Ref(bifucation_weight), branch.solutions, Ref(θ))
end
