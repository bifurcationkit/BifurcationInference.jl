################################################################################# loss gradient
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::AbstractVector; kwargs...)

	∇marginals = sum( branch->∇marginal_likelihood(F,branch,θ,targets;kwargs...), branches )
	marginals = sum( branch->marginal_likelihood(F,branch,θ,targets;kwargs...), branches )

	∇norms = sum( branch->∇normalisation(F,branch,θ), branches )
	norms = sum( branch->normalisation(F,branch,θ), branches )

	return log(norms)-log(marginals), ∇norms/norms -∇marginals/marginals
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
	∂z = jacobian( z -> -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * integrand(F,z,θ), z )
	θi = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
	return tr.( getindex.( Ref(∂z), θi, : ) ) # div(z) = tr(∂z) for each component θ
end

################################################################################
function ∇likelihood( F::Function, z::BorderedArray, θ::AbstractVector, targets::AbstractVector; kwargs...)
	integrand(F,z,θ) = likelihood(F,z,θ,targets; kwargs...)
	return gradient(θ->integrand(F,z,θ),θ) + ∇region(F,integrand,z,θ)
end

function ∇marginal_likelihood( F::Function, branch::Branch, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return branch.ds'∇likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); kwargs...)
end

function ∇normalisation( F::Function, branch::Branch, θ::AbstractVector)
	return gradient( θ -> normalisation(F,branch,θ), θ ) + branch.ds'∇region.( Ref(F), Ref(bifucation_weight), branch.solutions, Ref(θ))
end

############################################################# autodiff wrappers for BorderedArray
import ForwardDiff: gradient,jacobian,hessian

function gradient( f, z::BorderedArray )
	return gradient( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function jacobian( f, z::BorderedArray )
	return jacobian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function hessian( f, z::BorderedArray )
	return hessian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end
