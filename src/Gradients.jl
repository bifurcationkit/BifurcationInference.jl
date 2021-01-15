################################################################################# loss gradient
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)
	if sum( branch->sum(s->s.bif,branch), branches ) ≥ 2length(targets.targets)

		L,ω = likelihood(F,branches,θ,targets;kwargs...), weight(F,branches,θ,targets;kwargs...)
		∇L,∇ω = ∇likelihood(F,branches,θ,targets;kwargs...), ∇weight(F,branches,θ,targets;kwargs...)

		return -log(L) + log(ω), -∇L/L + ∇ω/ω 
	else

		K = curvature(F,branches,θ,targets;kwargs...)
		∇K = ∇curvature(F,branches,θ,targets;kwargs...)

		return -log(K), -∇K/K
	end
end

################################################################################
struct Gradient <: Function
	f::Function
	integrand::Integrand
end
(f::Gradient)(args...;kwargs...) = f.f(args...;kwargs...)

∇likelihood = Gradient( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; kwargs...)
	∇likelihood = ForwardDiff.gradient(θ->likelihood(F,z,θ,targets; kwargs...),θ) + deformation(F,z,θ)'ForwardDiff.gradient(z->likelihood(F,z,θ,targets; kwargs...),z)
	return ∇likelihood + likelihood(F,z,θ,targets; kwargs...)*∇region(F,z,θ)
end, likelihood)

∇weight = Gradient(function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; kwargs...)
	∇weight = ForwardDiff.gradient(θ->weight(F,z,θ,targets; kwargs...),θ) + deformation(F,z,θ)'ForwardDiff.gradient(z->weight(F,z,θ,targets; kwargs...),z)
	return ∇weight + weight(F,z,θ,targets; kwargs...)*∇region(F,z,θ)
end, weight)

∇curvature = Gradient(function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; kwargs...)
	∇curvature = ForwardDiff.gradient(θ->curvature(F,z,θ,targets; kwargs...),θ) + deformation(F,z,θ)'ForwardDiff.gradient(z->curvature(F,z,θ,targets; kwargs...),z)
	return ∇curvature + curvature(F,z,θ,targets; kwargs...)*∇region(F,z,θ)
end, curvature)

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
	∇deformation = reshape( ForwardDiff.jacobian(z->deformation(F,z,θ),z), length(z),length(θ),length(z) )
	tangent = tangent_field(F,z,θ)

	∇region = similar(θ)
	for k ∈ 1:length(θ)
		∇region[k] = tangent'∇deformation[:,k,:]tangent
	end

	return ∇region
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

############################################################# autodiff wrappers for BorderedArray
import ForwardDiff

function ForwardDiff.gradient( f, z::BorderedArray )
	return ForwardDiff.gradient( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ForwardDiff.jacobian( f, z::BorderedArray )
	return ForwardDiff.jacobian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end

function ForwardDiff.hessian( f, z::BorderedArray )
	return ForwardDiff.hessian( z -> f( BorderedArray(z[Not(end)],z[end]) ), [z.u; z.p] )
end
