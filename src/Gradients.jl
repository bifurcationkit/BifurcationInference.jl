################################################################################# loss gradient
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::AbstractVector; kwargs...)
	if sum( branch->sum(branch.bifurcations), branches ) ≥ 2length(targets)

		∇marginals = sum( branch->∇marginal_likelihood(F,branch,θ,targets;kwargs...), branches )
		marginals = sum( branch->marginal_likelihood(F,branch,θ,targets;kwargs...), branches )

		∇norms = sum( branch->∇normalisation(F,branch,θ), branches )
		norms = sum( branch->normalisation(F,branch,θ), branches )

		return log(norms)-log(marginals), ∇norms/norms -∇marginals/marginals
	else

		∇curvatures = sum( branch->∇curvature(F,branch,θ), branches )
		curvatures = sum( branch->curvature(F,branch,θ), branches )

		return -log(curvatures), -∇curvatures/curvatures
	end
end

########################################################################### jacobians

# statespace jacobian
function ∂Fu(F::Function,z::BorderedArray,θ::AbstractVector)
	return ForwardDiff.jacobian( u -> F(   u, (θ=θ,p=z.p) ), z.u )
end

# parameter jacobian
function ∂Fθ(F::Function,z::BorderedArray,θ::AbstractVector)
	return ForwardDiff.jacobian( θ -> F( z.u, (θ=θ,p=z.p) ), θ )
end

# augmented jacobian
function ∂Fz(F::Function,z::BorderedArray,θ::AbstractVector)
	return ForwardDiff.jacobian( z -> F( z[Not(end)], (θ=θ,p=z[end]) ), [z.u; z.p] )
end

################################### gradient terms due to changing integration region dz
function ∇region( F::Function, integrand::Function, z::BorderedArray, θ::AbstractVector)
	∂z = ForwardDiff.jacobian( z -> -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * integrand(F,z,θ), z )
	θi = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
	return tr.( getindex.( Ref(∂z), θi, : ) ) # div(z) = tr(∂z) for each component θ
end

################################################################################
function ∇likelihood( F::Function, z::BorderedArray, θ::AbstractVector, targets::AbstractVector; kwargs...)
	integrand(F,z,θ) = likelihood(F,z,θ,targets; kwargs...)
	return ForwardDiff.gradient(θ->integrand(F,z,θ),θ) + ∇region(F,integrand,z,θ)
end

function ∇marginal_likelihood( F::Function, branch::Branch, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return branch.ds'∇likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); kwargs...)
end

function ∇normalisation( F::Function, branch::Branch, θ::AbstractVector)
	return ForwardDiff.gradient( θ -> normalisation(F,branch,θ), θ ) + branch.ds'∇region.( Ref(F), Ref(bifucation_weight), branch.solutions, Ref(θ))
end

###########################################################################
function ∇curvature(F::Function,z::BorderedArray,θ::AbstractVector)
	return ForwardDiff.gradient(θ->curvature(F,z,θ),θ) + ∇region(F,curvature,z,θ) #bottleneck
end

function ∇curvature(F::Function,branch::Branch,θ::AbstractVector)
	return branch.ds'∇curvature.( Ref(F), branch.solutions, Ref(θ) )
end

############################################################# autodiff wrappers for BorderedArray
import ForwardDiff,ReverseDiff

function ForwardDiff.gradient( f, z::BorderedArray )
	return ForwardDiff.gradient( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ForwardDiff.jacobian( f, z::BorderedArray )
	return ForwardDiff.jacobian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ForwardDiff.hessian( f, z::BorderedArray )
	return ForwardDiff.hessian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ReverseDiff.gradient( f, z::BorderedArray )
	return ReverseDiff.gradient( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ReverseDiff.jacobian( f, z::BorderedArray )
	return ReverseDiff.jacobian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ReverseDiff.hessian( f, z::BorderedArray )
	return ReverseDiff.hessian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end
