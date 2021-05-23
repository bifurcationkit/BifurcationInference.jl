################################################################################
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)

	predictions = unique([ s.z for branch ∈ branches for s ∈ branch if s.bif ], atol=3*step(targets.parameter) )
	λ = length(targets.targets)-length(predictions)

	if λ≠0 
		Φ,∇Φ = measure(F,branches,θ), ∇measure(F,branches,θ)
		return errors(predictions,targets) - λ*log(Φ), ∇errors(F,predictions,θ,targets) - λ*∇Φ/Φ
	else
		return errors(predictions,targets), ∇errors(F,predictions,θ,targets)
	end
end

################################################################################
function ∇errors( F::Function, predictions::AbstractVector{<:BorderedArray}, θ::AbstractVector, targets::StateSpace)
	return mean( p′-> mean( z->(z.p-p′)^2, predictions; type=:geometric )*mean( z-> 2∂p(F,z,θ)/(z.p-p′), predictions; type=:arithmetic ), targets.targets; type=:arithmetic )
end

function ∇measure( F::Function, z::BorderedArray, θ::AbstractVector )
	return gradient(θ->measure(F,z,θ),θ) + deformation(F,z,θ)'gradient(z->measure(F,z,θ),z) + measure(F,z,θ)*∇region(F,z,θ)
end

###########################################################################
function ∇measure( F::Function, branch::Branch, θ::AbstractVector )
	return sum( s -> ∇measure(F,s.z,θ)*s.ds, branch )
end

function ∇measure( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector )
	return sum( branch -> ∇measure(F,branch,θ), branches )
end

############################################## gradient term due to changing integration region dz
deformation( F::Function, z::BorderedArray, θ::AbstractVector ) = -∂Fz(F,z,θ)\∂Fθ(F,z,θ)
function ∇region( F::Function, z::BorderedArray, θ::AbstractVector )

	∇deformation = reshape( jacobian(z->deformation(F,z,θ),z), length(z),length(θ),length(z) )
	tangent = tangent_field(F,z,θ)

	∇region = similar(θ)
	for k ∈ 1:length(θ)
		∇region[k] = tangent'∇deformation[:,k,:]tangent
	end

	return ∇region
end

########################################################################### jacobians

# statespace jacobian
∂Fu(F::Function,z::BorderedArray,θ::AbstractVector) = ∂Fz(F,z,θ)[:,Not(end)]

# parameter jacobian
∂Fθ(F::Function,z::BorderedArray,θ::AbstractVector) = jacobian( θ -> F(z,θ), θ )

# augmented jacobian
∂Fz(F::Function,z::BorderedArray,θ::AbstractVector) = jacobian( z -> F(z,θ), z )

############################################################# autodiff wrappers for BorderedArray
import ForwardDiff: gradient,jacobian

function gradient( f, z::BorderedArray )
	return gradient( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function jacobian( f, z::BorderedArray )
	return jacobian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end