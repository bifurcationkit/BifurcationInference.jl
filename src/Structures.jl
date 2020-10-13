import Base: length,show,push!
struct Branch{V<:AbstractVector,T<:Number}

    solutions::Vector{BorderedArray{V,T}}
    eigvals::Vector{Vector{Complex{T}}}
    ds::Vector{T}

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
    bifurcations::Ref{<:AbstractVector{T}}
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
Branch(V::DataType,T::DataType) = Branch( BorderedArray{V,T}[], Vector{Complex{T}}[], T[] )
length(branch::Branch) = length(branch.solutions)
dim(branch::Branch) = length(first(branch.solutions).u)

function push!(branch::Branch,state::PALCStateVariables)
    push!(branch.solutions,copy(solution(state)))
    push!(branch.eigvals,state.eigvals)
    push!(branch.ds,state.ds)
end

# display methods
show(io::IO, branch::Branch{V,T}) where {V,T} = print(io,
    "Branch{$V,$T}[dim=$(dim(branch)) states=$(length(branch)), parameter=($(round(first(branch.solutions).p,sigdigits=3))->$(round(last(branch.solutions).p,sigdigits=3)))]")
show(io::IO, branches::Vector{Branch{V,T}}) where {V,T} = print(io,
    "Vector{Branches}[dim=$(dim(first(branches))) branches=$(length(branches)), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{V,T}}) where {V,T} = show(io,branches)
