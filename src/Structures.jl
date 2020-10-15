import Base: length,show,push!
struct Branch{V<:AbstractVector,T<:Number}

    solutions::Vector{BorderedArray{V,T}}
    eigvals::Vector{Vector{Complex{T}}}
    ds::Vector{T}

end

struct StateSpace{N,T}

    roots::AbstractVector{<:AbstractVector{<:AbstractVector}}
    parameter::AbstractRange
    targets::Ref{<:AbstractVector}

end

"""
	data = StateSpace( dimension, parameter, targets; nRoots=2, eltype=Float64 )

Define state space with targets to be used in optimisation

# Positional Arguments
- `dimension` dimensionality of state space `u`
- `parameter` one dimensional bifurcation parameter grid `p`
- `targets` vector of target locations

# Keyword Arguments
- `nRoots` number of roots to continue solutions from
- `eltype` numeric type for vector elements

# Struct Fields
- `roots` vector of roots `F(u.p)=0` to continue solutions from
- `parameter` one dimensional bifurcation parameter grid `p`
- `targets` vector of target locations
"""
function StateSpace(dimension::Integer,parameter::AbstractRange,targets::AbstractVector; nRoots::Integer=2, eltype::DataType=Float64)
    roots, nTargets = fill( [zero(SizedVector{dimension,eltype})] ,nRoots ), length(targets)
    return StateSpace{dimension,eltype}( SizedVector{nRoots}(roots), StepRangeLen{eltype}(parameter), Ref(SVector{nTargets,eltype}(targets)) )
end

"""
	branch = Branch(T)

Object to contain the accumulation of continuation results for one branch

# Fields
- `solutions` vector of steady state values `(u,p)` along branch
- `eigvals` vector of eigenvalues at each point
- `ds` vector of arclength steps sizes between continuation points
"""
Branch(V::DataType,T::DataType) = Branch( BorderedArray{V,T}[], Vector{Complex{T}}[], T[] )
length(branch::Branch) = length(branch.solutions)
dim(branch::Branch) = length(first(branch.solutions).u)

function push!(branch::Branch,state::PALCStateVariables)
    push!(branch.solutions,copy(solution(state)))
    push!(branch.eigvals,state.eigvals)
    push!(branch.ds,abs(state.ds))
end

# display methods
show(io::IO, branch::Branch{V,T}) where {V,T} = print(io,
    "Branch{$V,$T}[dim=$(dim(branch)) states=$(length(branch)), parameter=($(round(first(branch.solutions).p,sigdigits=3))->$(round(last(branch.solutions).p,sigdigits=3)))]")
show(io::IO, branches::Vector{Branch{V,T}}) where {V,T} = print(io,
    "Vector{Branches}[dim=$(dim(first(branches))) branches=$(length(branches)), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{V,T}}) where {V,T} = show(io,branches)
show(io::IO, states::StateSpace{N,T}) where {N,T} = print(io,"StateSpace{$N,$T}(parameters=$(states.parameter),targets=$(states.targets.x))")
