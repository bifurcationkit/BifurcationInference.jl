using PseudoArcLengthContinuation: AbstractLinearSolver, AbstractBorderedLinearSolver, AbstractEigenSolver, _axpy
using LinearAlgebra

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
