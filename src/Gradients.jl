################################################################################
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	predictions = unique([ s.z for branch ∈ branches for s ∈ branch if s.bif ], atol=3*step(targets.parameter) )

	if length(targets.targets)≠length(predictions)
		Φ,∇Φ = measure(F,branches,θ,targets), ∇measure(F,branches,θ,targets)
		S,∇S = length(F,branches,θ,targets), ∇length(F,branches,θ,targets)
		return errors(predictions,targets) - log(Φ) + log(S), ∇errors(F,predictions,θ,targets) - ∇Φ/Φ + ∇S/S
	else
		return errors(predictions,targets), ∇errors(F,predictions,θ,targets)
	end
end

################################################################################
function ∇errors( F::Function, predictions::AbstractVector{<:BorderedArray}, θ::AbstractVector, targets::StateSpace)
	return mean( p′-> mean( z->(z.p-p′)^2, predictions; type=:geometric )*mean( z-> 2velocity(F,z,θ)/(z.p-p′), predictions; type=:arithmetic ), targets.targets; type=:arithmetic )
end

function ∇measure( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; newtonOptions=NewtonPar(verbose=false,maxIter=800,tol=1e-6) )
	∂implicit, _, _ = newtonOptions.linsolver( -∂Fz(F,z,θ)', gradient(z->measure(F,z,θ,targets),z) )
	return gradient( θ -> measure(F,z,θ,targets) + F(z,θ)'∂implicit , θ ) + measure(F,z,θ,targets)*∇region(F,z,θ)
end

function ∇length( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; newtonOptions=NewtonPar(verbose=false,maxIter=800,tol=1e-6) )
	∂implicit, _, _ = newtonOptions.linsolver( -∂Fz(F,z,θ)', gradient(z->length(F,z,θ,targets),z) )
	return gradient( θ -> length(F,z,θ,targets) + F(z,θ)'∂implicit , θ ) + length(F,z,θ,targets)*∇region(F,z,θ)
end

###########################################################################
function ∇measure( F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace )
	return sum( s -> ∇measure(F,s.z,θ,targets)*s.ds, branch )
end

function ∇measure( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace )
	return sum( branch -> ∇measure(F,branch,θ,targets), branches )
end

function ∇length( F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace )
	return sum( s -> ∇length(F,s.z,θ,targets)*s.ds, branch )
end

function ∇length( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace )
	return sum( branch -> ∇length(F,branch,θ,targets), branches )
end

############################################## gradient term due to changing integration region dz
deformation( F::Function, z::BorderedArray, θ::AbstractVector ) = -∂Fz(F,z,θ)\∂Fθ(F,z,θ) # todo@(gszep) matrix inverse is computational bottleneck
function ∇region( F::Function, z::BorderedArray, θ::AbstractVector )

	∇deformation = reshape( jacobian(z->deformation(F,z,θ),z), length(z),length(θ),length(z) )
	tangent = tangent_field(F,z,θ)
	t = [tangent.u;tangent.p]

	∇region = similar(θ)
	for k ∈ 1:length(θ)
		∇region[k] = t'∇deformation[:,k,:]t
	end

	return ∇region
end

########################################################################### jacobians

# statespace jacobian
∂Fu(F::Function,z::BorderedArray,θ::AbstractVector) = ∂Fz(F,z,θ)[:,Not(end)]

# parameter jacobian
∂Fθ(F::Function,z::BorderedArray,θ::AbstractVector) = jacobian( θ->F(z,θ), θ )

# augmented jacobian
∂Fz(F::Function,z::BorderedArray,θ::AbstractVector) = jacobian( z->F(z,θ), z )

############################################################# autodiff wrappers for BorderedArray
import ForwardDiff: derivative,gradient,jacobian
derivative(f,x::BorderedArray,s::BorderedArray) = derivative( λ -> f(x + λ*s), 0.0 )

function gradient( f, z::BorderedArray )
	return gradient( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function jacobian( f, z::BorderedArray )
	return jacobian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

# jacobian( u->F(u,(θ=θ,p=z.p)), z.u )
# [∂Fu(F,z,θ) derivative( p->F(z.u,(θ=θ,p=p)), z.p )]
# return [ gradient( u -> f(BorderedArray(u,z.p)), z.u ); derivative( p -> f(BorderedArray(z.u,p)), z.p ) ]
# return [ jacobian( u -> f(BorderedArray(u,z.p)), z.u ) derivative( p -> f(BorderedArray(z.u,p)), z.p ) ]