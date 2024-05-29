################################################################################
function loss(F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)

    predictions = unique([s.z for branch ∈ branches for s ∈ branch if s.bif]; atol=2 * step(targets.parameter))
    λ = length(targets.targets) - length(predictions)

    if λ ≠ 0
        Φ = measure(F, branches, θ, targets)
        return errors(predictions, targets) - λ * log(Φ)
    else
        return errors(predictions, targets)
    end
end

################################################################################
function errors(predictions::AbstractVector{<:BorderedArray}, targets::StateSpace)
    return mean(p′ -> mean(z -> (z.p - p′)^2, predictions; type=:geometric), targets.targets; type=:arithmetic)
end

realeigvals(J) = real(eigen(J).values)
function measure(F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace)
    λ, dλ = realeigvals(∂Fu(F, z, θ)), derivative(z -> realeigvals(∂Fu(F, z, θ)), z, tangent_field(F, z, θ))
    return window_function(targets.parameter, z) * mapreduce((λ, dλ) -> 2 / (2 + abs(sinh(2λ) / dλ)), +, λ, dλ)
end

# Giles, M. 2008. An extended collection of matrix derivative results for forward and reverse mode automatic differentiation
# credit: @mateuszbaran https://github.com/JuliaManifolds/Manifolds.jl/pull/27#issuecomment-521693995
import LinearAlgebra: eigen

function make_eigen_dual(val::Real, partial)
    Dual{tagtype(partial)}(val, partial.partials)
end

function make_eigen_dual(val::Complex, partial::Complex)
    Complex(Dual{tagtype(real(partial))}(real(val), real(partial).partials),
        Dual{tagtype(imag(partial))}(imag(val), imag(partial).partials))
end

function eigen(A::StridedMatrix{Dual{T,V,N}}) where {T,V,N}

    Av = map(a -> a.value, A)
    λ, U = eigen(Av)

    dΛ = U \ A * U
    dλ = diag(dΛ)

    F = similar(Av, eltype(λ))
    for i ∈ axes(A, 1), j ∈ axes(A, 2)
        if i == j
            F[i, j] = 0
        else
            F[i, j] = inv(λ[j] - λ[i])
        end
    end
    dU = U * (F .* dΛ)

    for i ∈ eachindex(dU)
        dU[i] = make_eigen_dual(U[i], dU[i])
    end
    return Eigen(dλ, dU)
end

################################################################################
function measure(F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace)
    return sum(s -> measure(F, s.z, θ, targets) * s.ds, branch)
end

function measure(F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace)
    return sum(branch -> measure(F, branch, θ, targets), branches)
end

########################################################## bifurcation distance and velocity
distance(F::Function, z::BorderedArray, θ::AbstractVector) = [F(z, θ); det(∂Fu(F, z, θ))]
function velocity(F::Function, z::BorderedArray, θ::AbstractVector; newton_options=NewtonPar(verbose=false, max_iterations=800, tol=1e-6))
    ∂implicit, _, _ = newton_options.linsolver(-jacobian(z -> distance(F, z, θ), z)', [zero(z.u); one(z.p)])
    return gradient(θ -> distance(F, z, θ)'∂implicit, θ)
end

################################################################################
function tangent_field(F::Function, z::BorderedArray, θ::AbstractVector)
    field = kernel(∂Fz(F, z, θ); nullity=length(z.p))
    return norm(field)^(-1) * BorderedArray(field[Not(end)], field[end]) # unit tangent field
end

################################################################################
using SpecialFunctions: erf
function window_function(parameter::AbstractVector, z::BorderedArray; β::Real=10)
    p_min, p_max = extrema(parameter)
    return (1 + erf((β * (z.p - p_min) - 3) / √2)) / 2 * (1 - (1 + erf((β * (z.p - p_max) + 3) / √2)) / 2)
end