import Base: length,show,push!
struct StateSpace{N,T}

    roots::AbstractVector{<:AbstractVector{<:AbstractVector}}
    parameter::AbstractRange
    targets::AbstractVector

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
    return StateSpace{dimension,eltype}( SizedVector{nRoots}(roots), StepRangeLen{eltype}(parameter), SVector{nTargets,eltype}(targets) )
end

"""
	branch = Branch{V,T}()

Initialises vector of named tuples that contain the following fields

# Fields
- `z` steady state solutions `(u,p)` along branch
- `λ` vector of eigenvalues
- `ds` arclength steps sizes between continuation points
- `bif` boolean telling us if point is a bifurcation
"""
BranchPoint{V,T} = NamedTuple{(:z,:λ,:ds,:bif),Tuple{BorderedArray{V,T},Vector{Complex{T}},T,Bool}}
Branch{V,T} = Vector{BranchPoint{V,T}}
dim(branch::Branch) = length(branch) > 0 ? length(first(branch).z)-1 : 0

function push!(branch::Branch,state::ContState)
    z,∂z = copy(solution(state)), [state.tau.u;state.tau.p]
    push!(branch, ( z=z, λ=state.eigvals, ds=norm(∂z)*abs(state.ds), bif=detectBifucation(state) ))
end

# display methods
show(io::IO, branches::Vector{Branch{V,T}}) where {V,T} = print(io,
    "Vector{Branch}[dim=$(dim(first(branches))) bifurcations=$(sum(branch->sum(s->s.bif,branch),branches)) branches=$(length(branches)), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{V,T}}) where {V,T} = show(io,branches)
show(io::IO, states::StateSpace{N,T}) where {N,T} = print(io,"StateSpace{$N,$T}(parameters=$(states.parameter),targets=$(states.targets))")
