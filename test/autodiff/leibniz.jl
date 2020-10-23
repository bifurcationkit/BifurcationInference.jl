using InvertedIndices,LinearAlgebra,Plots
using ForwardDiff

try # compatibility with Julia 1.3
	using BifurcationKit
	global Iterable = BifurcationKit.ContIterable
catch
	using BifurcationKit
	global Iterable = BifurcationKit.PALCIterable
end

using Setfield: @lens
using Flux: σ

############################################################## F(z,θ) = 0 region definition
F( z::AbstractVector, θ::AbstractVector ) = F( z[Not(end)], z[end], θ )
function F( u::AbstractVector, p::Number, θ::AbstractVector )

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))
	β = 1/3.1

	# p+θ[1]*u[1]-u[1]^3 + θ[2]
	F[1] = p - (1-exp(β*( θ[1]*u[1]-u[1]^3 )))/β + θ[2]
	#F[2] = u[1] - u[2] #3θ[2] / (1+u[1]^2) - u[2]

	return F
end

u = [1.0] # initial root to peform continuation from
W(p;β=10) = (1-σ(β*(p-2)))*σ(β*(p+2))

# integrand under dz
integrand( z::AbstractVector, θ::AbstractVector ) = integrand( z[Not(end)], z[end], θ )
function integrand( u::AbstractVector, p::Number, θ::AbstractVector )
	return W(p)
end

########################################################## gradients
function ∇integrand( z::AbstractVector, θ::AbstractVector )

	# application of the general leibniz rule
	return ForwardDiff.gradient(θ->integrand(z,θ),θ) + ∇region(z,θ)
end

################################### gradient terms due to changing integration region dz
function ∇region( z::AbstractVector, θ::AbstractVector )
	return [ tr( ForwardDiff.jacobian( z -> velocity(z,θ,k) * integrand(z,θ), z )) for k ∈ 1:length(θ) ]
end
velocity( z::AbstractVector, θ::AbstractVector, k) = -∂Fz(z,θ)\∂Fθ(z,θ,k)

########################################################
cost(θ1, θ2; ds=0.001) = cost((θ1,θ2); ds=ds)
function cost(θ;ds=0.001)
	θ = typeof(θ) <: Tuple ? [θ...] : θ
	try
		∂S = solutions(θ,ds=ds)

		# plot(size=(500,500),ylabel="states, u",xlabel="parameter, p")
		# plot!(map(z->z[end],∂S), map(z->z[Uidx],∂S ),lw=3,label="",color=:darkblue)
		# plot!(map(z->z[end],∂S₊),map(z->z[Uidx],∂S₊),lw=3,label="",color=:lightblue)|>display

		return sum(z->integrand(z,θ),∂S)*ds

	catch
		println("cost failed for θ = $θ")
		return NaN
	end
end

∇cost(θ1, θ2; ds=0.001) = ∇cost((θ1,θ2); ds=ds)
function ∇cost(θ;ds=0.001)
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
		maxSteps = 10000, pMin = -3.0, pMax = 3.0, saveSolEveryStep = 0,
		newtonOptions = newtonOptions
	)

	# we define an iterator to hold the continuation routine
	J(u,p) = ForwardDiff.jacobian(x->F(x,p,θ),u)
	v, _, converged, _ = newton( (u,p) -> F(u,p,θ), J, u, -2.99, newtonOptions)
	if converged u .= v else throw("newton not converged") end

	iter = Iterable( (u,p) -> F(u,p,θ), J,
		u, -2.99, (@lens _), options; verbosity = 0)

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
∂Fθ(z::AbstractVector,θ::AbstractVector,k) = ∂Fθ(z,θ)[:,k]

# augmented statespace jacobian
function ∂Fz( z::AbstractVector, θ::AbstractVector )
	return ForwardDiff.jacobian( z -> F(z,θ) , z )
end

function tangent_field( z::AbstractVector, θ::AbstractVector)

	∂F = ∂Fz(z,θ) # construct tangent field T(z) := det[ ẑ , ∂Fz ]
	field = [ (-1)^(zi+1) * det(∂F[:,Not(zi)]) for zi ∈ 1:length(z) ] # cofactor expansion
	return field / norm(field) # unit tangent field
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

	x,y = range(-1/2,0,length=2), range(0,1/2,length=2)
	plot(size=(600,600), xlabel="parameters, θ")
	contour!( x, y, cost, alpha=0.5 )

	x,y = range(-1/2,0,length=2), range(0,1/2,length=2)
	grid = collect(Iterators.product(x,y))
	ϵ = 1e-6

	xGrid = vcat(map(x->x[1], grid)...)
	yGrid = vcat(map(x->x[2], grid)...)

	gradients = map( central_differences, xGrid, yGrid )
	quiver!( xGrid, yGrid, quiver=(ϵ*first.(gradients),ϵ*last.(gradients)),
		color=:darkblue, lw=3, alpha=2norm.(gradients) )

	gradients = map( ∇cost, xGrid, yGrid )
	println(gradients)
	quiver!( xGrid, yGrid, quiver=(ϵ*first.(gradients),ϵ*last.(gradients)),
		color=:gold, lw=2 )

	plot!([],[],color=:gold, lw=3, label="ForwardDiff")
	plot!([],[],color=:darkblue, lw=3, label="Central Differences") |> display

end
unit_test()

function plot_field(θ; Δθ = 0.5, θidx = 2, ds=0.001)

	∂S =  solutions(θ,ds=ds)
	∂S₊ = solutions(θ+Δθ*OneHot(length(θ),θidx),ds=ds)
	divergence = map(z->∇region(z,θ)[θidx],∂S )

	plot(size=(500,500),ylabel="states, u",xlabel="parameter, p", xlim=(-2,2), ylim=(-1,6))
	for Uidx ∈ 1:1

		plot!(map(z->z[end],∂S), Uidx.+map(z->z[Uidx],∂S ),label="",color=:darkblue)
		plot!(map(z->z[end],∂S), Uidx.+map(z->z[Uidx],∂S ),lw=10divergence,label="",color=:darkblue)
		plot!(map(z->z[end],∂S), Uidx.+map(z->z[Uidx],∂S ),lw=-10divergence,label="",color=:darkred)
		plot!(map(z->z[end],∂S₊),Uidx.+map(z->z[Uidx],∂S₊),lw=1,label="",color=:lightblue)

		∂Sdownsampled = ∂S[1:100:end]
		Vu = map(z->velocity(z,θ,θidx)[Uidx],∂Sdownsampled)
		Vp = map(z->velocity(z,θ,θidx)[end],∂Sdownsampled)
		norms = map(z->norm(velocity(z,θ,θidx)),∂Sdownsampled)

		quiver!( map(z->z[end],∂Sdownsampled), Uidx.+map(z->z[Uidx],∂Sdownsampled),
			quiver=(Δθ*Vp,Δθ*Vu), color=:darkblue, alpha=norms)
		end
	plot!()|>display
	println("total divergence = $(sum(divergence)*ds)")
	@assert(abs(sum(sum( z->velocity(z,θ)'tangent_field(z,θ), ∂S)))<1e-12)
end

sum(divergence)*0.001

u = [1.0,1.0]
z = ∂S[2640]
θ = [-1,1]

sum( z->velocity(z,θ,1)'tangent_field(z,θ),
	solutions(θ,ds=0.001) )

ForwardDiff.jacobian( θ -> velocity(z,θ,1)/norm(velocity(z,θ,1)), θ )
ForwardDiff.jacobian( θ -> velocity(z,θ,1), θ )
ForwardDiff.jacobian( θ -> velocity(z,θ), θ )


∂Fz(z,[1,1])
∂Fz(z,[1,1])[2,:]

# using formula for velocities of implicit regions
z = ∂S[3520]
velocity(z,θ)[:,θidx]

ForwardDiff.jacobian( z -> velocity(z,θ)[:,θidx], z )
ForwardDiff.jacobian( z -> velocity(z,θ)[:,2], z )

∂z = ForwardDiff.jacobian( z -> velocity(z,θ), z )


θi = [ (i-1)*length(z)+1:i*length(z) for i ∈ 1:length(θ) ]
tr.( getindex.( Ref(∂z), θi, : ) ) # div(z) = tr(∂z) for each component θ
