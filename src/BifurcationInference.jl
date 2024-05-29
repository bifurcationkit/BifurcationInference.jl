module BifurcationInference

using BifurcationKit: BifurcationProblem, re_make, PALC, ContIterable, newton, ContinuationPar, NewtonPar, DeflationOperator
using BifurcationKit: BorderedArray, AbstractLinearSolver, AbstractEigenSolver, BorderingBLS
using BifurcationKit: ContState, detect_bifurcation

using ForwardDiff: Dual, tagtype, derivative, gradient, jacobian
using Flux: Momentum, update!

using Setfield: @lens, @set, setproperties
using Parameters: @unpack

using InvertedIndices: Not
using LinearAlgebra, StaticArrays

include("Structures.jl")
include("Utils.jl")

include("Objectives.jl")
include("Gradients.jl")
include("Plots.jl")

export plot, @unpack, BorderedArray, SizedVector
export StateSpace, deflationContinuation, train!
export getParameters, loss, ∇loss, norm

""" root finding with newton deflation method"""
function findRoots!(f::Function, J::Function, roots::AbstractVector{<:AbstractVector},
    parameters::NamedTuple, hyperparameters::ContinuationPar;
    maxRoots::Int=3, max_iterations::Int=500, verbosity=0)

    hyperparameters = @set hyperparameters.newton_options = setproperties(
        hyperparameters.newton_options; max_iterations=max_iterations, verbose=verbosity)

    # search for roots across parameter range
    pRange = range(hyperparameters.p_min, hyperparameters.p_max, length=length(roots))
    roots .= findRoots.(Ref(f), Ref(J), roots, pRange, Ref(parameters), Ref(hyperparameters); maxRoots=maxRoots)
end

function findRoots(f::Function, J::Function, roots::AbstractVector{V}, p::T,
    parameters::NamedTuple, hyperparameters::ContinuationPar{T,S,E}; maxRoots::Int=3, converged=false
) where {T<:Number,V<:AbstractVector{T},S<:AbstractLinearSolver,E<:AbstractEigenSolver}

    Zero = zero(first(roots))
    inf = Zero .+ Inf

    # search for roots at specific parameter value
    deflation = DeflationOperator(one(T), dot, one(T), [inf]) # dummy deflation at infinity
    parameters = @set parameters.p = p

    problem = BifurcationProblem(f, roots[begin] .+ hyperparameters.ds, parameters; J=J)
    solution = newton(problem, deflation, hyperparameters.newton_options)

    for u ∈ roots # update existing roots
        solution = newton(re_make(problem; u0=u .+ hyperparameters.ds), deflation, hyperparameters.newton_options)

        i = 0
        while any(isnan.(solution.residuals)) & (i < hyperparameters.newton_options.max_iterations)
            u .= randn(length(u))

            solution = newton(re_make(problem; u0=u .+ hyperparameters.ds), deflation, hyperparameters.newton_options)
            i += 1
        end

        @assert(!any(isnan.(solution.residuals)), "f(u,p) = $(solution.residuals[end]) at u = $(solution.u), p = $(parameters.p), θ = $(parameters.θ)")
        if solution.converged
            push!(deflation, solution.u)
        else
            break
        end
    end

    u = Zero
    if solution.converged || length(deflation) == 1 # search for new roots
        while length(deflation) - 1 < maxRoots

            solution = newton(re_make(problem; u0=u .+ hyperparameters.ds), deflation, hyperparameters.newton_options)

            # make sure new roots are different from existing
            if any(isapprox.(Ref(solution.u), deflation.roots, atol=2 * hyperparameters.ds))
                break
            end
            if solution.converged
                push!(deflation, solution.u)
            else
                break
            end
        end
    end

    filter!(root -> root ≠ inf, deflation.roots) # remove dummy deflation at infinity
    @assert(length(deflation.roots) > 0, "No roots f(u,p)=0 found at p = $(parameters.p), θ = $(parameters.θ); try increasing max_iterations")
    return deflation.roots
end

""" deflation continuation method """
function deflationContinuation(f::Function, roots::AbstractVector{<:AbstractVector{V}},
    parameters::NamedTuple, hyperparameters::ContinuationPar{T,S,E};
    maxRoots::Int=3, max_iterations::Int=500, resolution=400, verbosity=0, kwargs...
) where {T<:Number,V<:AbstractVector{T},S<:AbstractLinearSolver,E<:AbstractEigenSolver}

    max_iterationsContinuation, ds = hyperparameters.newton_options.max_iterations, hyperparameters.ds
    J(u, p) = jacobian(x -> f(x, p), u)

    findRoots!(f, J, roots, parameters, hyperparameters; maxRoots=maxRoots, max_iterations=max_iterations, verbosity=verbosity)
    pRange = range(hyperparameters.p_min, hyperparameters.p_max, length=length(roots))
    intervals = ([zero(T), step(pRange)], [-step(pRange), zero(T)])

    branches = Vector{Branch{V,T}}()
    problem = BifurcationProblem(f, roots[begin][begin], parameters, (@lens _.p); J=J)

    hyperparameters = @set hyperparameters.newton_options.max_iterations = max_iterationsContinuation
    linsolver = BorderingBLS(hyperparameters.newton_options.linsolver)
    algorithm = PALC()

    for (i, us) ∈ enumerate(roots)
        for u ∈ us # perform continuation for each root

            # forwards and backwards branches
            for (p_min, p_max) ∈ intervals

                hyperparameters = setproperties(hyperparameters;
                    p_min=pRange[i] + p_min, p_max=pRange[i] + p_max,
                    ds=sign(hyperparameters.ds) * ds)

                # main continuation method
                branch = Branch{V,T}()
                parameters = @set parameters.p = pRange[i] + hyperparameters.ds

                try
                    iterator = ContIterable(re_make(problem; u0=u, params=parameters), algorithm, hyperparameters; verbosity=verbosity)
                    for state ∈ iterator
                        push!(branch, state)
                    end

                    midpoint = sum(s -> s.z.p, branch) / length(branch)
                    if minimum(pRange) < midpoint < maximum(pRange)
                        push!(branches, p_min < 0 ? reverse(branch) : branch)
                    end

                catch error
                    printstyled(color=:red, "Continuation Error at f(u,p)=$(f(u,parameters))\nu=$u, p=$(parameters.p), θ=$(parameters.θ)\n")
                    rethrow(error)
                end
                hyperparameters = @set hyperparameters.ds = -hyperparameters.ds
            end
        end
    end

    hyperparameters = @set hyperparameters.ds = ds
    updateParameters!(hyperparameters, branches; resolution=resolution)
    return unique(branches; atol=10 * hyperparameters.ds)
end
end # module
