############################################# hyperparameter updates
function getParameters(data::StateDensity{T}; maxIter::Int=10, tol=1e-12) where {T<:Number}
    return ContinuationPar(

        pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),

            newtonOptions = NewtonPar( linsolver=LinearSolver(), eigsolver=EigenSolver(),
            verbose=false,maxIter=maxIter,tol=tol),

        detectFold = false, detectBifurcation = true)
end
@nograd getParameters

function updateParameters(parameters::ContinuationPar{T, S, E}, steady_states::Vector{Branch{T}};
    resolution=400 ) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

    # estimate scale from steady state curves
    branch_points = map(length,steady_states)
    ds = maximum(branch_points)*parameters.ds/resolution
    parameters = setproperties(parameters;ds=ds,dsmin=ds,dsmax=ds)

    return parameters
end
@nograd updateParameters

################################################## differentiable solvers
# reference: https://github.com/FluxML/Zygote.jl/pull/327
import LinearAlgebra: eigen
@adjoint function eigen(A::AbstractMatrix)
    eV = eigen(A)
    e,V = eV
    n = size(A,1)
    eV, function (Δ)
        Δe, ΔV = Δ
        if ΔV === nothing
			(inv(V)'*Diagonal(Δe)*V', )
        elseif Δe === nothing
			F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
			(inv(V)'*(F .* (V'ΔV))*V', )
        else
			F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
			(inv(V)'*(Diagonal(Δe) + F .* (V'ΔV))*V', )
        end
    end
end

############################ non-mutating solvers
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

############################################################## plotting
import Plots: plot
function plot(steady_states::Vector{Branch{T}}, data::StateDensity{T}; idx::Int=1) where {T<:Number}
	right_axis = plot(steady_states; idx=idx, displayPlot=false)

	vline!( data.bifurcations, label="", color=:gold)
	plot!( right_axis,[],[], color=:gold, legend=:bottomleft,
        alpha=1.0, label=L"\mathrm{targets}\,\,\mathcal{D}") |> display
end

function plot(steady_states::Vector{Branch{T}}; idx::Int=1, displayPlot=true) where {T<:Number}

	plot([NaN],[NaN],label="",xlabel=L"\mathrm{parameter},\,p",
		right_margin=20mm,size=(500,400))
	right_axis = twinx()

    for branch in steady_states

        stability = map( λ -> all(real(λ).<0), branch.eigvals)
        determinants = map( λ -> prod(real(λ)), branch.eigvals)

        plot!(branch.parameter, map(x->x[idx],branch.state), linewidth=2, alpha=0.5, label="", grid=false,
            ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(z)=0",
            color=map( stable -> stable ? :darkblue : :lightblue, stability )
            )

        plot!(right_axis, branch.parameter, determinants, linewidth=2, alpha=0.5, label="", grid=false,
        	ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(z)",
            color=map( stable -> stable ? :red : :pink, stability )
        	)

        scatter!( branch.parameter[branch.bifurcations],
				  map(x->x[idx],branch.state)[branch.bifurcations],
            label="", m = (3.0,3.0,:black,stroke(0,:none)))
    end

	plot!(right_axis,[],[], color=:darkblue, legend=:bottomleft, linewidth=2,
        alpha=1.0, label=L"\mathrm{steady\,states}")
	scatter!(right_axis,[],[], label=L"\mathrm{prediction}\,\,\mathcal{P}(\theta)", legend=:bottomleft,
        m = (1.0, 1.0, :black, stroke(0, :none)))

	if displayPlot
		plot!(right_axis,[],[], color=:red, legend=:bottomleft,
			alpha=1.0, label=L"\mathrm{determinant}", dpi=500, linewidth=2) |> display
	else
		plot!(right_axis,[],[], color=:red, legend=:bottomleft,
			alpha=1.0, label=L"\mathrm{determinant}", dpi=500, linewidth=2)

		return right_axis
	end
end
@nograd plot
