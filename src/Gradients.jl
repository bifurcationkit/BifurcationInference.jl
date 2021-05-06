################################################################################# loss gradient
function ∇loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)

	pmin,pmax = extrema(targets.parameter)
	N,M = length(targets.targets), reduce(+, [ pmin ≤ s.z.p ≤ pmax for branch ∈ branches for s ∈ branch if s.bif ]; init=0)

	ω,∇ω = weight(F,branches,θ,targets;kwargs...) + eps(), ∇weight(F,branches,θ,targets;kwargs...)

	L =  sum( p -> exp( errors(F,branches,θ,targets;p=p,kwargs...)/ω )                                                                                                          , targets.targets ) / N
	∇L = sum( p -> exp( errors(F,branches,θ,targets;p=p,kwargs...)/ω ) * ( ∇errors(F,branches,θ,targets;p=p,kwargs...)*ω - errors(F,branches,θ,targets;p=p,kwargs...)*∇ω ) / ω^2, targets.targets ) / N

	K = curvature(F,branches,θ,targets;kwargs...)
	∇K = ∇curvature(F,branches,θ,targets;kwargs...)

	return L + (2N-M)/(1+abs(K)), ∇L - sign(K)*(2N-M)/(1+abs(K))^2 * ∇K
end

################################################################################
struct Gradient <: Function
	f::Function
	integrand::Integrand
end
(f::Gradient)(args...;kwargs...) = f.f(args...;kwargs...)

∇errors = Gradient( function( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace; p=0.0, kwargs...)
	∇errors = ForwardDiff.gradient(θ->errors(F,z,θ,targets;p=p,kwargs...),θ) + deformation(F,z,θ)'ForwardDiff.gradient(z->errors(F,z,θ,targets;p=p,kwargs...),z)
	return ∇errors + errors(F,z,θ,targets;p=p,kwargs...)*∇region(F,z,θ)
end, errors)

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
