using PseudoArcLengthContinuation, SparseArrays, LinearAlgebra, Plots, Setfield, Parameters
using PseudoArcLengthContinuation: AbstractLinearSolver,AbstractBorderedLinearSolver, _axpy
const PALC = PseudoArcLengthContinuation

using Zygote: Buffer, @nograd, @adjoint, @adjoint!
using Flux: gradient, params

############################ define non-mutating solvers
struct EigenSolver <: AbstractEigenSolver end
function (l::EigenSolver)(J, nev::Int64)
	F = eigen(Array(J))
	return Complex.(F.values), Complex.(F.vectors), true, 1
end

struct LinearSolver <: AbstractLinearSolver end
function (l::LinearSolver)(J, rhs; a₀ = 0, a₁ = 1, kwargs...)
	return _axpy(J, a₀, a₁) \ rhs, true, 1
end
function (l::LinearSolver)(J, rhs1, rhs2; a₀ = 0, a₁ = 1, kwargs...)
	return J \ rhs1, J \ rhs2, true, (1, 1)
end

@with_kw struct BorderedLinearSolver{S<:AbstractLinearSolver} <: AbstractBorderedLinearSolver
	solver::S = LinearSolver()
end
function (lbs::BorderedLinearSolver{S})( J, dR, dzu, dzp::T, R, n::T,
		xiu::T = T(1), xip::T = T(1); shift::Ts = nothing)  where {T, S, Ts}

	# we make this branching to avoid applying a zero shift
	if isnothing(shift)
		x1, x2, _, (it1, it2) = lbs.solver(J, R, dR)
	else
		x1, x2, _, (it1, it2) = lbs.solver(J, R, dR; a₀ = shift)
	end

	dl = (n - dot(dzu, x1) * xiu) / (dzp * xip - dot(dzu, x2) * xiu)
	x1 = x1 .- dl .* x2

	return x1, dl, true, (it1, it2)
end

######################### define model and continuation method
k,θ = [3.0],[1.0]
F = (x, p) -> (@. p + x - (θ+x)^(k+1)/(k+1))
J = (x, p) -> 1 .- (θ+x)[:,:].^k

function steady_states() global opts,iter

	opts = PALC.ContinuationPar(dsmax = 0.1, dsmin = 1e-3, ds = -0.001, maxSteps = 530, pMin = -3., pMax = 3.,
		saveSolEveryNsteps = 0, computeEigenValues = true, detectBifurcation = true,
		newtonOptions = NewtonPar(tol = 1e-8, verbose = false, maxIter=5000,
			linsolver=LinearSolver(), eigsolver=EigenSolver(), linesearch=false)
	)
	iter = PALC.PALCIterable(F, J, [3.0], 1.0, (@lens _), opts, BorderedLinearSolver(); verbosity=0)

	resp = Buffer(Float64[])
	resx = Buffer(Float64[])

	for state in iter
		push!(resx, getx(state)[1])
		push!(resp, getp(state))
	end

	return copy(resp), copy(resx)
end

######################### providing inplace gradients where possible


# @adjoint! function copyto!(xs::Union{AbstractArray,BorderedArray}, ys::Union{AbstractArray,BorderedArray})
# 	xs_ = copy(xs)
# 	copyto!(xs, ys), function (dxs)
# 		copyto!(xs_, xs)
# 		return (nothing, dxs)
# 	end
# end
#
# @adjoint function rmul!(a, b)
#     c = rmul!(a, b)
#     return c, function(Δ)
#         a = rmul!(a, 1/b)
#         return (Δ*b,a*Δ)
#     end
# end
#

#
# @adjoint function axpy!(a, x, y)
#     c = axpy!(a, x, y)
#     return c, function(Δ)
# 		y = minus!(y,a*x)
#         return (Δ*x,a*Δ,Δ)
#     end
# end



# zygote not compatible with @error logging https://github.com/FluxML/Zygote.jl/issues/386
# differenitated kwargs bug https://github.com/FluxML/Zygote.jl/issues/584


######################### minimal test

θ,k=[-1.0],[2.0];
plot(steady_states()...; label = "", xlabel = "p")
dθ = gradient(params(θ)) do
	p,u = steady_states()
	sum(p.^2+u.^2)
end
	dθ[θ][1]

function evaluate_gradient(x; index=1) global θ
	copyto!(θ,[x...])

	dθ = gradient(params(θ)) do
		p,u = steady_states()
		sum(p.^2+u.^2)
	end

	p,u = steady_states()
	return [ sum(p.^2+u.^2), dθ[θ][index] ]
end

ϕ = range(-1,1,length=500)
	gradients = hcat(evaluate_gradient.(ϕ)...)
	L,dL = gradients[1,:], gradients[2,:]
	finite_differences = vcat(diff(L)/step(ϕ),NaN)

	plot(ϕ,L,label="",color="black",ylabel="Function")
	right_axis = twinx(); plot(right_axis,label="",
		ylabel="Gradient",
		legend=:bottomleft)

	plot!(right_axis,ϕ,finite_differences,label="Finite Differences",color="gold",fillrange=0,alpha=0.5)
	plot!(right_axis,ϕ,dL,label="Zygote AutoDiff",color="lightblue",fillrange=0,alpha=0.6) |> display
