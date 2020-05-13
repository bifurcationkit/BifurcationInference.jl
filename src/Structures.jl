struct StateDensity{T}
    parameter::StepRangeLen{T}
    bifurcations::Vector{T}
end

struct Branch{T}

    state::Vector{Vector{T}}
    parameter::Vector{T}

    bifurcations::Vector{NamedTuple{(:state,:parameter),Tuple{Vector{T},T}}}
    eigvals::Vector{Vector{Complex{T}}}
end

import Base: length,show
length(branch::Branch) = length(branch.parameter)
Branch(T::DataType) = Branch(Vector{T}[], T[], NamedTuple{(:state,:parameter),Tuple{Vector{T},T}}[], Vector{Complex{T}}[])

show(io::IO, branch::Branch{T}) where T = print(io,
    "Branch{$T}[bifurcations=$(length(branch.bifurcations)), states=$(length(branch))]")
show(io::IO, branches::Vector{Branch{T}}) where T = print(io,
    "Branches{$T}[length=$(length(branches)), bifurcations=$(sum(branch->length(branch.bifurcations),branches)), states=$(sum(branch->length(branch),branches))]")
show(io::IO, M::MIME"text/plain", branches::Vector{Branch{T}}) where T = show(io,branches)
