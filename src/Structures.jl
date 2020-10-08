import Base: length,show
@with_kw struct Branch{T<:Number}

    state::Vector{Vector{T}}
    parameter::Vector{T}
    ds::Vector{T}

    bifurcations::Vector{Bool}
    eigvals::Vector{Vector{Complex{T}}}
    # dim::UInt = length(first(state))
end

"""
	data = StateDensity(parameter,bifurcations)

Returns target bifurcation data to be used in optimisation such as `loss(steady_states::Vector{Branch{T}}, data::StateDensity{T}, curvature::Function)`

# Arguments
- `parameter` bifurcation parameter grid to use
- `bifurcations` vector of bifurcation locations

"""
struct StateDensity{T<:Number}
    parameter::StepRangeLen{T}
    bifurcations::Ref{<:Vector{T}}
end

"""
	branch = Branch(T)

Object to contain the accumulation of continuation results for one branch

# Fields
- `state` vector of steady state values along branch
- `parameter` vector of parameter values along branch
- `ds` vector of arclength steps sizes between continuation points
- `bifurcations` vector locations where an eigenvalue crosses zero
- `eigvals` vector of eigenvalues at each point
"""
Branch(T::DataType) = Branch(Vector{T}[], T[], T[], Bool[], Vector{Complex{T}}[])
length(branch::Branch) = length(branch.parameter)

# display methods
show(io::IO, branch::Branch{T}) where T = print(io,
    "Branch{$T}[bifurcations=$(sum(branch.bifurcations)), states=$(length(branch)), parameter=($(round(branch.parameter[1],sigdigits=3))->$(round(branch.parameter[end],sigdigits=3)))]")
show(io::IO, branches::Vector{Branch{T}}) where T = print(io,
    "Vector{Branch{$T}}[branches=$(length(branches)), bifurcations=$(sum(branch->sum(branch.bifurcations),branches)÷2), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{T}}) where T = show(io,branches)