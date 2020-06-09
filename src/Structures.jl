import Base: length,show,copy
struct Branch{T<:Number}

    state::Vector{Vector{T}}
    parameter::Vector{T}
    ds::Vector{T}

    bifurcations::Vector{Bool}
    eigvals::Vector{Vector{Complex{T}}}
end

struct CuBranch{T<:Number}
	state::CuArray{T}
    parameter::CuArray{T}
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
    bifurcations::CuArray
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
length(branch::Union{Branch,CuBranch}) = length(branch.parameter)

# display methods
show(io::IO, branch::Branch{T}) where T = print(io,
    "Branch{$T}[bifurcations=$(sum(branch.bifurcations)), states=$(length(branch)), parameter=($(round(branch.parameter[1],sigdigits=3))->$(round(branch.parameter[end],sigdigits=3)))]")
show(io::IO, branch::CuBranch{T}) where T = print(io,
    "CuBranch{$T}[states=$(length(branch)), parameter=($(round(minimum(branch.parameter),sigdigits=3))->$(round(maximum(branch.parameter),sigdigits=3)))]")
show(io::IO, branches::Vector{Branch{T}}) where T = print(io,
    "Vector{Branch{$T}}[branches=$(length(branches)), bifurcations=$(sum(branch->sum(branch.bifurcations),branches)÷2), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{T}}) where T = show(io,branches)

struct BranchBuffer{T<:Number}

    state::Buffer{Vector{T},Vector{Vector{T}}}
    parameter::Buffer{T,Vector{T}}
    ds::Buffer{T,Vector{T}}

    bifurcations::Buffer{Bool,Vector{Bool}}
    eigvals::Buffer{Vector{Complex{T}},Vector{Vector{Complex{T}}}}
end

"""Zygote compatible version of `Branch` object"""
BranchBuffer(T::DataType) = BranchBuffer( Buffer(Vector{T}[]), Buffer(T[]), Buffer(T[]), Buffer(Bool[]), Buffer(Vector{Complex{T}}[]))

# coverting branch object fields from zygote buffers to arrays
function copy(branch::BranchBuffer)

	state = copy(branch.state)
	parameter = copy(branch.parameter)
	ds = copy(branch.ds)

	bifurcations = copy(branch.bifurcations)
	eigvals = copy(branch.eigvals)

	return Branch(state,parameter,ds,bifurcations,eigvals)
end
copy(branch::Branch) = branch
