################################################################################
function loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)

	pmin,pmax = extrema(targets.parameter)
	if any( p -> pmin ≤ p ≤ pmax, [ s.z.p for branch ∈ branches for s ∈ branch if s.bif ])

		return -log(likelihood(F,branches,θ,targets;kwargs...)) + log(weight(F,branches,θ,targets;kwargs...))
	else
		return -log(curvature(F,branches,θ,targets;kwargs...))
	end
end

################################################################################
struct Integrand <: Function f::Function end
(f::Integrand)(args...;kwargs...) = f.f(args...;kwargs...)

likelihood = Integrand( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; kwargs...)
	return gaussian_mixture(targets,z; kwargs...) * weight(F,z,θ,targets; kwargs...)
end)

weight = Integrand( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; α::Real=1e3, kwargs... )
	return exp( -α*det(∂Fu(F,z,θ))^2 )
end)

curvature = Integrand( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; kwargs... )

	∂det = ForwardDiff.gradient(z->det(∂Fu(F,z,θ)),z)
	∂∂det = ForwardDiff.hessian(z->det(∂Fu(F,z,θ)),z)

	# tangent field basis
	Jz = ForwardDiff.jacobian(z->tangent_field(F,z,θ),z)
	Tz = tangent_field(F,z,θ)

	return (Tz'∂∂det*Tz + ∂det'Jz*Tz)^2
end)

################################################################################
function gaussian_mixture( targets::StateSpace, z::BorderedArray; ϵ::Real=0.1, kwargs...)
	return sum( p->exp( -(p-z.p)^2/ϵ ), targets.targets ) / length(targets.targets) / √(ϵ*π)
end

################################################################################
function (integrand::Integrand)( F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace; kwargs...)
	return sum( s -> window_function( targets.parameter, s.z; kwargs... )*integrand( F, s.z, θ, targets; kwargs...)s.ds, branch )
end

function (integrand::Integrand)( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	return sum( branch -> integrand( F, branch, θ, targets; kwargs...), branches )
end

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