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
    z,∂z = copy(state.z_old), [state.tau.u;state.tau.p]
    push!(branch, ( z=z, λ=state.eigvals, ds=norm(∂z)*abs(state.ds), bif=detectBifucation(state) ))
end

import Base: zero,isapprox,unique,+ ################################## methods for BorderedArray
import LinearAlgebra: norm

+(x::BorderedArray, y::BorderedArray) = BorderedArray(x.u+y.u,x.p+y.p)
norm(x::Branch, y::Branch) = mapreduce( (x,y)->norm(x.z.u-y.z.u)+norm(x.z.p-y.z.p),+,x,y)/min(length(x),length(y))

zero(::Type{BorderedArray{T,U}}; ϵ::Bool=true) where {T<:Union{Number,StaticArray},U<:Union{Number,StaticArray}} = BorderedArray(zero(T).+ϵ*eps(),zero(U).+ϵ*eps())
isapprox( x::BorderedArray, y::BorderedArray; kwargs... ) = isapprox( [x.u;x.p], [y.u;y.p] ; kwargs...)
function unique(X::AbstractVector{T}; kwargs...) where T<:BorderedArray
    z = T[]
    for x ∈ X 
        if all( zi -> ~isapprox( x, zi; kwargs... ), z )
            push!(z,x)
        end
    end
    return z
end

function unique(X::AbstractVector{T}; kwargs...) where T<:Branch
    branches = T[]
    for branch ∈ X 
        if all( branchᵢ -> ~isapprox( norm(branch,branchᵢ), 0; kwargs... ), branches )
            push!(branches,branch)
        end
    end
    return branches
end

function kernel(A::AbstractMatrix; nullity::Int=0)

    if iszero(nullity) # default to SVD
        return nullspace(A)

    else 
        return qr(A').Q[:,end+1-nullity:end]
    end
end

#################################################### display methods
show(io::IO, branches::Vector{Branch{V,T}}) where {V,T} = print(io,
    "Vector{Branch}[dim=$(dim(first(branches))) bifurcations=$(length([ s.z for branch ∈ branches for s ∈ branch if s.bif ])) branches=$(length(branches)), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{V,T}}) where {V,T} = show(io,branches)
show(io::IO, states::StateSpace{N,T}) where {N,T} = print(io,"StateSpace{$N,$T}(parameters=$(states.parameter),targets=$(states.targets))")