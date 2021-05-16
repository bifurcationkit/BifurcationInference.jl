################################################################################
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	pmin,pmax = extrema(targets.parameter)
	predictions = sum([ s.bif/2 for branch ∈ branches for s ∈ branch if (pmin ≤ s.z.p ≤ pmax) ])

	Φ,∇Φ = measure(F,branches,θ,targets), ∇measure(F,branches,θ,targets)
	return errors(branches,targets) - (length(targets.targets)-predictions)*log(Φ), ∇errors(F,branches,θ,targets) - (length(targets.targets)-predictions)*∇Φ/Φ
end

################################################################################
function ∇errors( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace)
	pmin,pmax = extrema(targets.parameter)

	predictions = [ s.z for branch ∈ branches for s ∈ branch if s.bif & (pmin ≤ s.z.p ≤ pmax) ]
	return mean( p′-> mean( z->(z.p-p′)^2, predictions; type=:geometric )*mean( z-> 2∂p(F,z,θ)/(z.p-p′), predictions; type=:arithmetic ), targets.targets; type=:arithmetic )
end

∇measure = Gradient( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; p=0.0, kwargs...)
	∇measure = gradient(θ->measure(F,z,θ,targets;p=p,kwargs...),θ) + deformation(F,z,θ)'gradient(z->measure(F,z,θ,targets;p=p,kwargs...),z)
	return ∇measure + measure(F,z,θ,targets;p=p,kwargs...)*∇region(F,z,θ)
end, measure)

###########################################################################
function (gradient::Gradient)( F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace; kwargs...)
	boundary_term = sum( s->deformation(F,s.z,θ)[end,:]*gradient.integrand( F, s.z, θ, targets; kwargs...)*boundaries( targets.parameter, s.z; kwargs... )*s.ds, branch )
	return sum( s -> window_function( targets.parameter, s.z; kwargs... )*gradient( F, s.z, θ, targets; kwargs...)s.ds, branch ) + boundary_term
end

function (gradient::Gradient)( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	return sum( branch -> gradient( F, branch, θ, targets; kwargs...), branches )
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