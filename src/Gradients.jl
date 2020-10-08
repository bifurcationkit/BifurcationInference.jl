########################################################################### jacobians

# statespace jacobian
function ∂Fu(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	return jacobian( u -> F( u,           (θ=θ,p=z[end]) ), z[Not(end)] )
end

# parameter jacobian
function ∂Fθ(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	return jacobian( θ -> F( z[Not(end)], (θ=θ,p=z[end]) ), θ           )
end

# augmented jacobian
function ∂Fz(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	return jacobian( z -> F( z[Not(end)], (θ=θ,p=z[end]) ), z           )
end

########################################################################### tangent field
# field tangent to bifurcation curve of F(z,θ)
function tangent_field(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}

	∂F = ∂Fz(F,z,θ) # augmented jacobian
	f = first(z)*first(θ)
	field = similar(z,typeof(f))

	for i ∈ 1:length(field)

		# construct tangent field T(z) := det[ ̂z , ∂Fz ]
		field[i] = det(∂F[:,Not(i)])
	end

	# unit tangent field
	return field / norm(field)
end

################################################################################# determinant curvature K(z|θ)
function determinant(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	return det(∂Fu(F,z,θ))
end

# determinant gradient field
function determinant_field(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	return gradient( z -> determinant(F,z,θ), z )
end

# determinant curvature wrt tangent field
function curvature(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	tangent = tangent_field(F,z,θ)
	return gradient( z -> determinant_field(F,z,θ)'tangent, z )'tangent
end

# gradient of curvature wrt θ
function ∇curvature(F::Function,z::AbstractVector{T},θ::AbstractVector{U}) where {T<:Number,U<:Number}
	return gradient( θ -> curvature(F,z,θ), θ )
end

################################################################################# likelihood L(z|θ,D)
function likelihood(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
		 ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
	errors = (targets.-z[end]).^2
	return exp(-determinant(F,z,θ)^2) * sum(exp.(-errors/ϵ)) / (π*length(targets)*√ϵ)
end

# likelihood corrected with unsupervised curvature term
function semisupervised_likelihood(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
	     λ::V=0.0, ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
	return likelihood(F,z,θ,targets; ϵ=ϵ) + λ*curvature(F,z,θ)^2
end

# gradients wrt θ
function ∇likelihood(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
	     ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
	return gradient( θ -> likelihood(F,z,θ,targets; ϵ=ϵ), θ )
end

function ∇semisupervised_likelihood(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
	     λ::V=0.0, ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
	return gradient( θ -> semisupervised_likelihood(F,z,θ,targets; λ=λ, ϵ=ϵ), θ )
end

################################################################################# region velocity
function ∇region(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
		 λ::V=0.0, ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}

	Jz = jacobian( z -> -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * semisupervised_likelihood(F,z,θ,targets; λ=λ, ϵ=ϵ) , z )
	idx = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
	return [ tr(Jz[θ,:]) for θ ∈ idx ] # divergence for each component θ
end

################################################################################# loss function L(θ|D)
function loss(F::Ref{<:Function},steady_states::AbstractVector{X},θ::Ref{<:AbstractVector{U}},targets::Ref{<:AbstractVector{T}};
	     λ::T=0.0, ϵ::T=0.1) where {T<:Number,U<:Number,X<:Branch{T}}

	# construct augmented state space z = (u,p)
	z  = vcat(map( branch -> map( vcat, branch.state, branch.parameter ), steady_states)...)
	ds = vcat(map( branch -> abs.(branch.ds), steady_states)...) # arclength between points

	return -log(sum( semisupervised_likelihood.(F,z,θ,targets; λ=λ, ϵ=ϵ) .*ds ))
end

# gradient of loss wrt θ
function ∇loss(F::Ref{<:Function},steady_states::AbstractVector{X},θ::Ref{<:AbstractVector{U}},targets::Ref{<:AbstractVector{T}};
	     λ::T=0.0, ϵ::T=0.1) where {T<:Number,U<:Number,X<:Branch{T}}
	
	# construct augmented state space z = (u,p)
	z  = vcat(map( branch -> map( vcat, branch.state, branch.parameter ), steady_states)...)
	ds = vcat(map( branch -> abs.(branch.ds), steady_states)...) # arclength between points

	L = loss(F,steady_states,θ,targets; λ=λ, ϵ=ϵ)
	∇L = -exp(L) * sum( ( ∇region.(F,z,θ,targets; λ=λ, ϵ=ϵ) .+ ∇semisupervised_likelihood.(F,z,θ,targets; λ=λ, ϵ=ϵ) ).*ds )
	return L, ∇L
end