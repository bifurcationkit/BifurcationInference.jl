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
	return norm(θ)^2 + 3exp(-(p-1)^2/0.05) # just an example
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
function cost(θ)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	∂S = solutions(θ)

	return sum(z->integrand(z,θ),∂S) / length(∂S)
end

function ∇cost(θ;scale=1e-3)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	∂S = solutions(θ)

	gradient = sum(z->∇integrand(z,θ),∂S) / length(∂S)
	return scale * gradient / norm(gradient)
end

########################################################### utils
function solutions(θ)
	θ = typeof(θ) <: Tuple ? [θ...] : θ

	# parameters for the continuation
	opts = ContinuationPar(dsmax = 0.01, dsmin = 0.01, ds = 0.01,
		maxSteps = 1000, pMin = 0.0, pMax = 2.0, saveSolEveryStep = 0,
		newtonOptions = NewtonPar(tol = 1e-8,verbose = false)
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


function finite_differences(x,y;scale=1e-3)

	grid = collect(Iterators.product(x,y))
	cost_landscape = cost.(grid)

	Px = vcat(map( x->x[1], grid[1:end-1,1:end-1])...)
	Py = vcat(map( x->x[2], grid[1:end-1,1:end-1])...)

	Vx = vcat(diff(cost_landscape, dims=1)[:,1:end-1]...)
	Vy = vcat(diff(cost_landscape, dims=2)[1:end-1,:]...)

	Nx = map( (x,y)->scale*x/norm([x,y]), Vx,Vy)
	Ny = map( (x,y)->scale*y/norm([x,y]), Vx,Vy)

	return Px,Py,Nx,Ny
end

###################################
###################################
###################################

######################### finite difference gradient estimate
x,y = range(1.4,2,length=31), range(0.1,0.4,length=29)
Px,Py,Vx,Vy = finite_differences(x,y)

##################################
##################################
################################## plot it all

plot(size=(600,600), xlabel="parameters, θ")
contourf!( x, y, (x,y)->cost((x,y)) )
plot!( x, maximum(y)*ones(length(x)), label="", fillrange=minimum(y), color=:white, alpha=0.5 )

quiver!( x[1:end-1], y[1:end-1]', quiver=(x,y)->∇cost((x,y)), color=:darkblue, lw=3)
quiver!( Px, Py, quiver=(Vx,Vy), lw=2, color=:gold)

plot!([],[],color=:darkblue, lw=3, label="ForwardDiff")
plot!([],[],color=:gold, lw=3, label="Finite Differences")
