################################################################################
function loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	pmin,pmax = extrema(targets.parameter)

	predictions = sum([ s.bif/2 for branch ∈ branches for s ∈ branch if (pmin ≤ s.z.p ≤ pmax) ])
	return errors(branches,targets) - (length(targets.targets)-predictions)*log(measure(F,branches,θ,targets))
end

################################################################################
function errors( branches::AbstractVector{<:Branch}, targets::StateSpace)
	pmin,pmax = extrema(targets.parameter)

	predictions = [ s.z for branch ∈ branches for s ∈ branch if s.bif & (pmin ≤ s.z.p ≤ pmax) ]
	return mean( p′-> mean( z->(z.p-p′)^2, predictions; type=:geometric ), targets.targets; type=:arithmetic )
end

measure = Integrand( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; kwargs... )
	return 1 / ( 1 + abs( det(F,z,θ) / ∂det(F,z,θ)'tangent_field(F,z,θ) ) )
end)

################################################################################
function (integrand::Integrand)( F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace; kwargs...)
	return sum( s -> window_function( targets.parameter, s.z; kwargs... )*integrand( F, s.z, θ, targets; kwargs...)s.ds, branch )
end

function (integrand::Integrand)( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	return sum( branch -> integrand( F, branch, θ, targets; kwargs...), branches )
end

########################################################################### determinant
import LinearAlgebra: det
det(F::Function,z::BorderedArray,θ::AbstractVector) = det(∂Fu(F,z,θ))
∂det(F::Function,z::BorderedArray,θ::AbstractVector) = gradient(z->det(F,z,θ),z)

########################################################## bifurcation distance and velocity
distance(F::Function,z::BorderedArray,θ::AbstractVector) = [ F(z,θ); det(F,z,θ) ]
velocity( F::Function, z::BorderedArray, θ::AbstractVector ) = - jacobian(z->distance(F,z,θ),z) \ jacobian(θ->distance(F,z,θ),θ)
∂p( F::Function, z::BorderedArray, θ::AbstractVector ) = velocity(F,z,θ)[end,:]

################################################################################
function tangent_field( F::Function, z::BorderedArray, θ::AbstractVector )

	∂F = ∂Fz(F,z,θ)
	field = similar(∂F[1,:])

	for i ∈ 1:length(z) # construct tangent field T(z) := det[ ẑ , ∂Fz ]
		field[i] = (-1)^(i+1) * det(∂F[:,Not(i)]) # laplace expansion of det
	end

	return field / norm(field) # unit tangent field
end

################################################################################
function window_function( parameter::AbstractVector, z::BorderedArray; β::Real=1e16, kwargs... )
	pMin,pMax = extrema(parameter)
	return σ(z.p-pMin;β=β) * ( 1 - σ(z.p-pMax;β=β) )
end

function boundaries( parameter::AbstractVector, z::BorderedArray; β::Real=1e16, kwargs... )
	pMin,pMax = extrema(parameter)
	return ℕ(z.p-pMin;β=β) - ℕ(z.p-pMax;β=β)
end

using SpecialFunctions: erf
function σ(x;β=10)
	return ( 1 + erf(β*x/√2) )/2
end

function ℕ(x;β=10)
	return β*exp(-(β*x)^2/2)/√(2π)
end