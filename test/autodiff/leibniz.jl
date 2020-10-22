using InvertedIndices,LinearAlgebra,Plots
using BifurcationKit,ForwardDiff
using Setfield: @lens

try # compatibility with Julia 1.3
	global Iterable = BifurcationKit.ContIterable
catch
	global Iterable = BifurcationKit.PALCIterable
end

############################################################## F(z,θ) = 0 region definition
F( z::AbstractVector, θ::AbstractVector ) = F( z[Not(end)], z[end], θ )
function F( u::AbstractVector, p::Number, θ::AbstractVector )

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	#F[1] = 1 / ( 1 + (p*u[2])^2 ) - θ[1]*u[1]
	#F[2] = 1 / ( 1 + (3*u[1])^2 ) - θ[2]*u[2]/10
	F[1] = p + θ[1]*u[1] + θ[2]*u[1]^3
	F[2] = u[1] - u[2] # dummy second dimension

	return F
end
u = [0.0,0.0] # initial root to peform continuation from

# integrand under dz
integrand( z::AbstractVector, θ::AbstractVector ) = integrand( z[Not(end)], z[end], θ )
function integrand( u::AbstractVector, p::Number, θ::AbstractVector )
	return 1.0
end

########################################################## gradients
function ∇integrand( z::AbstractVector, θ::AbstractVector )

	# application of the general leibniz rule
	return ForwardDiff.gradient(θ->integrand(z,θ),θ) + ∇region(z,θ)
end

################################### gradient terms due to changing integration region dz
function ∇region( z::AbstractVector, θ::AbstractVector )

	# using formula for velocities of implicit regions
	∂z = ForwardDiff.jacobian( z -> velocity(z,θ) * integrand(z,θ), z )

	θi = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
	return tr.( getindex.( Ref(∂z), θi, : ) ) # div(z) = tr(∂z) for each component θ
end
velocity( z::AbstractVector, θ::AbstractVector) = -∂Fz(z,θ)\∂Fθ(z,θ)

########################################################
cost(θ1, θ2; ds=0.01) = cost((θ1,θ2); ds=ds)
function cost(θ;ds=0.01)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	try
		∂S = solutions(θ,ds=ds)
		return sum(z->integrand(z,θ),∂S)*ds

	catch
		println("cost failed for θ = $θ")
		return NaN
	end
end

∇cost(θ1, θ2; ds=0.01) = ∇cost((θ1,θ2); ds=ds)
function ∇cost(θ;ds=0.01)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	try
		∂S = solutions(θ,ds=ds)
		return sum(z->∇integrand(z,θ),∂S)*ds

	catch
		println("∇cost failed for θ = $θ")
		return NaN*zero(θ)
	end
end

########################################################### utils
solutions(θ1, θ2; ds=0.01) = solutions((θ1,θ2); ds=ds)
function solutions(θ;ds=0.01)
	θ = typeof(θ) <: Tuple ? [θ...] : θ

	# parameters for the continuation
	newtonOptions = NewtonPar(tol = 1e-5, verbose = false)
	options = ContinuationPar(dsmax = ds, dsmin = ds, ds = ds,
		maxSteps = 1000, pMin = -2.1, pMax = 2.0, saveSolEveryStep = 0,
		newtonOptions = newtonOptions
	)

	# we define an iterator to hold the continuation routine
	J(u,p) = ForwardDiff.jacobian(x->F(x,p,θ),u)
	v, _, converged, _ = newton( (u,p) -> F(u,p,θ), J, u, -2.0, newtonOptions)
	if converged u .= v else throw("newton not converged") end

	iter = Iterable( (u,p) -> F(u,p,θ), J,
		u, -2.0, (@lens _), options; verbosity = 0)

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

central_differences(θ1, θ2; Δθ=1e-2) = central_differences((θ1,θ2); Δθ=Δθ)
function central_differences(θ;Δθ=1e-2)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	gradient = similar(θ)

	f = cost(θ)
	for i ∈ 1:length(θ)
		d = OneHot(length(θ),i)

		Δf₊, Δf₋ = cost(θ+Δθ*d), cost(θ-Δθ*d)
		gradient[i] = (Δf₊-Δf₋)/(2Δθ)
	end
	return gradient
end

##################################
##################################
##################################
function unit_test()

	x,y = range(-3/2,-1/2,length=50), range(1,2,length=50)
	plot(size=(600,600), xlabel="parameters, θ")
	contour!( x, y, cost, alpha=0.5 )

	x,y = range(-3/2,-1/2,length=20), range(1,2,length=20)
	grid = collect(Iterators.product(x,y))
	ϵ = 1e-6

	xGrid = vcat(map(x->x[1], grid)...)
	yGrid = vcat(map(x->x[2], grid)...)

	gradients = map( central_differences, xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*first.(gradients),ϵ*last.(gradients)),
		color=:darkblue, lw=3, alpha=norm.(gradients) )

	gradients = map( ∇cost, xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*first.(gradients),ϵ*last.(gradients)),
		color=:gold, lw=2, alpha=norm.(gradients) )

	plot!([],[],color=:gold, lw=3, label="ForwardDiff")
	plot!([],[],color=:darkblue, lw=3, label="Central Differences") |> display

end
unit_test()
