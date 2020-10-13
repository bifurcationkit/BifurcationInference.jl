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
function ∂Fz(F::Function,z::BorderedArray,θ::AbstractVector) # allocation needed?
	return jacobian( z -> F( z[Not(end)], (θ=θ,p=z[end]) ), [z.u; z.p] )
end

################################################################################# bifurcation density
function bifucation_density(F::Function,z::BorderedArray,θ::AbstractVector)
	return exp( -det(∂Fu(F,z,θ))^2 ) / √π
end

# ################################################################################# likelihood
function likelihood( F::Function, z::BorderedArray, θ::AbstractVector,
		 targets::AbstractVector; λ=0.0, ϵ=0.1)

	errors = ( targets .- z.p ) .^ 2
	mixture = sum(exp.(-errors/ϵ)) / (length(targets)*√(ϵ*π))
	return bifucation_density(F,z,θ) * mixture
end

function ∇likelihood( F::Function, z::BorderedArray, θ::AbstractVector,
		 targets::AbstractVector; λ=0.0, ϵ=0.1)

	# term due to changing integration region
	∂region = jacobian( z -> ∇region(F,z,θ,targets;λ=λ,ϵ=ϵ), [z.u; z.p] )
	idx = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]

	# divergence for each component θ
	∂region = [ tr(∂region[i,:]) for i ∈ idx ]
	return gradient( θ -> likelihood(F,z,θ,targets;λ=λ,ϵ=ϵ), θ ) .+ ∂region
end

function ∇region( F::Function, z::AbstractVector, θ::AbstractVector,
		 targets::AbstractVector; λ=0.0, ϵ=0.1)

	z = BorderedArray(z[Not(end)],z[end])
	return -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * likelihood(F,z,θ,targets;λ=λ,ϵ=ϵ)
end

function likelihood( F::Function, branch::Branch, θ::AbstractVector,
	     targets::AbstractVector; λ=0.0, ϵ=0.1)

	integrand = likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); λ=λ, ϵ=ϵ)
	return abs.(branch.ds)'integrand
end

function ∇likelihood( F::Function, branch::Branch, θ::AbstractVector,
		 targets::AbstractVector; λ=0.0, ϵ=0.1)

	∇integrand = ∇likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); λ=λ, ϵ=ϵ)
	return abs.(branch.ds)'∇integrand
end

################################################################################# loss function L(θ|D)
function loss( F::Ref{<:Function}, branches::AbstractVector{<:Branch}, θ::Ref{<:AbstractVector},
	     targets::Ref{<:AbstractVector}; λ=0.0, ϵ=0.1)

	return -log(sum( likelihood.(F,branches,θ,targets;λ=λ,ϵ=ϵ) ) )
end

# gradient of loss wrt θ
function ∇loss( F::Ref{<:Function}, branches::AbstractVector{<:Branch}, θ::Ref{<:AbstractVector},
	     targets::Ref{<:AbstractVector}; λ=0.0, ϵ=0.1)

	L = loss(F,branches,θ,targets;λ=λ,ϵ=ϵ)
	∇L = -exp(L) * sum( ∇likelihood.(F,branches,θ,targets;λ=λ,ϵ=ϵ) )
	return L, ∇L
end

########################################################################### tangent field
# field tangent to bifurcation curve of F(z,θ)
# function tangent_field(F::Function,z::BorderedArray,θ::AbstractVector)
#
# 	∂F = ∂Fz(F,z,θ) # augmented jacobian
# 	f = first(z)*first(θ)
# 	field = zeros(typeof(f),length(z))
#
# 	for i ∈ 1:length(field)
#
# 		# construct tangent field T(z) := det[ ̂z , ∂Fz ]
# 		field[i] = det(∂F[:,Not(i)])
# 	end
#
# 	# unit tangent field
# 	return field / norm(field)
# end


#
# function ∇likelihood( F::Function, branch::Branch,
# 	     θ::AbstractVector, targets::AbstractVector; λ=0.0, ϵ=0.1)
# 	return gradient( θ -> likelihood(F,branch,θ,targets; λ=λ, ϵ=ϵ), θ ) .+ ∇region(F,branch,θ,targets; λ=λ, ϵ=ϵ)
# end
#
# function ∇region( F::Function, branch::Branch,
# 	     θ::AbstractVector,targets::AbstractVector; λ=0.0, ϵ=0.1)
#
# 	Jz = jacobian( z -> -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * likelihood(F,branch,θ,targets; λ=λ, ϵ=ϵ) , z )
# 	idx = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
# 	return [ tr(Jz[θ,:]) for θ ∈ idx ] # divergence for each component θ
# end

# ( ∇region.(F,z,θ,targets; λ=λ, ϵ=ϵ) .+ ∇semisupervised_likelihood.(F,z,θ,targets; λ=λ, ϵ=ϵ) ).*ds

# gradients wrt θ
# function ∇likelihood( F::Function, branch::Branch,
# 		 θ::AbstractVector, targets::AbstractVector; λ=0.0, ϵ=0.1)
#
# 	parameters = map( z -> z.p, branch.solutions)
#  	determinants = map( λ -> prod(real(λ)), branch.eigvals)
#  	errors = (parameters.-targets').^2
#
#  	∂determinants = map( λ -> prod(real(λ)), branch.eigvals)
#
# 	∂integrand = -2determinants .* ∂determinants .* exp.(-determinants.^2) .* exp.(-errors/ϵ)
# 	return sum( abs.(branch.ds)'∂integrand ) / (π*length(targets)*√ϵ)
# end


# function ∇likelihood(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
# 	     ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
# 	return gradient( θ -> likelihood(F,z,θ,targets; ϵ=ϵ), θ )
# end
#
# function ∇semisupervised_likelihood(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
# 	     λ::V=0.0, ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
# 	return gradient( θ -> semisupervised_likelihood(F,z,θ,targets; λ=λ, ϵ=ϵ), θ )
# end

# ################################################################################# region velocity
# function ∇region(F::Function,z::AbstractVector{T},θ::AbstractVector{U},targets::AbstractVector{V};
# 		 λ::V=0.0, ϵ::V=0.1) where {T<:Number,U<:Number,V<:Number}
#
# 	Jz = jacobian( z -> -∂Fz(F,z,θ)\∂Fθ(F,z,θ) * semisupervised_likelihood(F,z,θ,targets; λ=λ, ϵ=ϵ) , z )
# 	idx = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
# 	return [ tr(Jz[θ,:]) for θ ∈ idx ] # divergence for each component θ
# end



# function ∇loss(F::Ref{<:Function},steady_states::AbstractVector{<:Branch},
# 		 θ::Ref{<:AbstractVector{T}},targets::Ref{<:AbstractVector{T}};
# 	     λ::T=0.0, ϵ::T=0.1) where T<:Number
#
# 	# construct augmented state space z = (u,p)
# 	z  = vcat(map( branch -> map( vcat, branch.state, branch.parameter ), steady_states)...)
# 	ds = vcat(map( branch -> abs.(branch.ds), steady_states)...) # arclength between points
#
# 	L = loss(F,steady_states,θ,targets; λ=λ, ϵ=ϵ)
# 	∇L = -exp(L) * sum( ( ∇region.(F,z,θ,targets; λ=λ, ϵ=ϵ) .+ ∇semisupervised_likelihood.(F,z,θ,targets; λ=λ, ϵ=ϵ) ).*ds )
# 	return L, ∇L
# end
#
# ################################################################ sum(det^2)^2=1 constraint using largrange method
# function regularisation(F::Ref{<:Function},steady_states::AbstractVector{X},θ::Ref{<:AbstractVector{U}},targets::Ref{<:AbstractVector{T}};
# 	     λ::T=0.0, ϵ::T=0.1) where {T<:Number,U<:Number,X<:Branch{T}}
#
# 	# construct augmented state space z = (u,p)
# 	z  = vcat(map( branch -> map( vcat, branch.state, branch.parameter ), steady_states)...)
# 	ds = vcat(map( branch -> abs.(branch.ds), steady_states)...) # arclength between points
#
# 	return -log(sum( semisupervised_likelihood.(F,z,θ,targets; λ=λ, ϵ=ϵ) .*ds ))
# end


#
# # determinant gradient field; cannot use ForwardDiff.gradient; github.com/JuliaDiff/ForwardDiff.jl/issues/197
# function determinant_field(F::Function,z::BorderedArray,θ::AbstractVector)
# 	return gradient( z -> determinant(F,z,θ), z )
# end
#
# # determinant curvature wrt tangent field
# function curvature(F::Function,z::BorderedArray,θ::AbstractVector)
# 	tangent = tangent_field(F,z,θ)
# 	return gradient( z -> determinant_field(F,z,θ)'tangent, z )'tangent
# end
#
# # gradient of curvature wrt θ
# function ∇curvature(F::Function,z::BorderedArray,θ::AbstractVector)
# 	return gradient( θ -> curvature(F,z,θ), θ )
# end
#
