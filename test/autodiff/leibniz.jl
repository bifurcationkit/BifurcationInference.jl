using InvertedIndices,LinearAlgebra,Plots
using BifurcationKit,ForwardDiff
using Setfield: @lens

############################################################## F(z,θ) = 0 region definition
F( z::AbstractVector, θ::AbstractVector ) = F( z[Not(end)], z[end], θ )
function F( u::AbstractVector, p::Number, θ::AbstractVector )

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = 1 / ( 1 + (p*u[2])^2 ) - θ[1]*u[1]
	F[2] = 1 / ( 1 + (3*u[1])^2 ) - θ[2]*u[2]

	return F
end

# integrand under dz
integrand( z::AbstractVector, θ::AbstractVector ) = integrand( z[Not(end)], z[end], θ )
function integrand( u::AbstractVector, p::Number, θ::AbstractVector )
	return 3exp(-(p-1)^2/0.01) # just an example
end

########################################################## gradients
function ∇integrand( z::AbstractVector, θ::AbstractVector )

	# application of the general leibniz rule
	return ForwardDiff.gradient(θ->integrand(z,θ),θ) + ∇region(z,θ)
end

################################### gradient terms due to changing integration region dz
function ∇region( z::AbstractVector, θ::AbstractVector )

	# using formula for velocities of implicit regions
	∂z = ForwardDiff.jacobian( z -> -∂Fz(z,θ)\∂Fθ(z,θ) * integrand(z,θ), z )

	∂z = reshape(∂z, length(z), length(z), length(θ))
	return [ tr(∂z[:,:,i]) for i ∈ 1:length(θ) ] # div(z) = tr(∂z) for each component θ
end

########################################################
function cost(θ;ds=0.01)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	∂S = solutions(θ,ds=ds)
	return sum(z->integrand(z,θ),∂S)*ds
end

function ∇cost(θ;ds=0.01)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	∂S = solutions(θ,ds=ds)
	return sum(z->∇integrand(z,θ),∂S)*ds
end

########################################################### utils
function solutions(θ;ds=0.01)
	θ = typeof(θ) <: Tuple ? [θ...] : θ

	# parameters for the continuation
	opts = ContinuationPar(dsmax = ds, dsmin = ds, ds = ds,
		maxSteps = 1000, pMin = 0.0, pMax = 2.0, saveSolEveryStep = 0,
		newtonOptions = NewtonPar(tol = 1e-8, verbose = false)
	)

	# we define an iterator to hold the continuation routine
	J(u,p) = ForwardDiff.jacobian(x->F(x,p,θ),u)
	iter = BifurcationKit.ContIterable( (u,p) -> F(u,p,θ), J,
		[0.0,0.0], 0., (@lens _), opts; verbosity = 0)

	solutions = Vector{Float64}[]
	for state in iter
		push!(solutions, [getx(state); getp(state)] )
	end
	return solutions
end

# parameter jacobian
function ∂Fθ( z::AbstractVector, θ::AbstractVector )
	return ForwardDiff.jacobian( θ -> F(z,θ), θ )
end

# augmented statespace jacobian
function ∂Fz( z::AbstractVector, θ::AbstractVector )
	return ForwardDiff.jacobian( z -> F(z,θ) , z )
end

##################################
##################################
################################## central differences

struct OneHot <: AbstractVector{Int}
	n::Int
	k::Int
end
import Base: size,getindex
size(x::OneHot) = (x.n,)
getindex(x::OneHot,i::Int) = Int(x.k==i)

function central_differences(θ;Δθ=1e-6)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	gradient = similar(θ)

	for i ∈ 1:length(θ)
		d = OneHot(length(θ),i)
		Δf₊, Δf₋ = cost(θ+Δθ*d), cost(θ-Δθ*d)

		gradient[i] = (Δf₊-Δf₋)/(2Δθ)
		@assert( (Δf₊+Δf₋)/2 ≈ cost(θ) )
	end
	return gradient
end

##################################
##################################
##################################

x,y = range(1.4,2,length=31), range(0.4,0.6,length=29)
plot(size=(600,600), xlabel="parameters, θ")

contourf!( x, y, (x,y)->cost((x,y)) )
plot!( x, maximum(y)*ones(length(x)), label="", fillrange=minimum(y), color=:white, alpha=0.5 )

quiver!( x, y',  quiver=(x,y)->1e-12*∇cost((x,y)), color=:darkblue, lw=3)
quiver!(  x, y', quiver=(x,y)->1e-12*central_differences((x,y)), color=:gold, lw=2)

plot!([],[],color=:darkblue, lw=3, label="ForwardDiff")
plot!([],[],color=:gold, lw=3, label="Central Differences")
