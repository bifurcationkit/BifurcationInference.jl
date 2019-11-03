using Flux.Tracker: Tracked,Call
using Flux: @back

struct TrackedComplex{T<:Complex} <: Real
  tracker::Tracked{T}
end

TrackedComplex(x::Complex) = TrackedComplex(Tracked(Call(nothing), x, zero(x)))

tracker(x::TrackedComplex) = x.tracker

track(f::Call, x::Complex) = TrackedComplex(Tracked(f, x, zero(x)))

back!(x::TrackedComplex) = back!(x, 1)

function Base.show(io::IO, x::TrackedComplex)
  show(io, data(x))
  print(io, " (tracked)")
end

Base.decompose(x::TrackedComplex) = Base.decompose(data(x))

Base.convert(::Type{TrackedComplex{T}}, x::TrackedComplex{T}) where T = x

Base.convert(::Type{TrackedComplex{T}}, x::TrackedComplex) where T =
  TrackedComplex(Tracked(x.tracker.f, convert(T, x.tracker.data)))

Base.convert(::Type{TrackedComplex{T}}, x::Complex) where T = TrackedComplex(convert(T, x))

Base.:(<)(x::TrackedComplex, y::TrackedComplex) = data(x) < data(y)
Base.:(==)(x::TrackedComplex, y::TrackedComplex) = data(x) == data(y)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedComplex) = Base.$f(data(x))
end

Base.Printf.fix_dec(x::TrackedComplex, n::Int) = Base.Printf.fix_dec(data(x), n)

Base.promote_rule(::Type{TrackedComplex{S}},::Type{T}) where {S,T} =
  TrackedComplex{promote_type(S,T)}

using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    $M.$f(a::TrackedComplex) = track($M.$f, a)
    back(::typeof($M.$f), Δ::Complex, a::TrackedComplex) =
      back(a, Δ * $(DiffRules.diffrule(M, f, :(data(a)))))
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :(data(a)), :(data(b)))
  @eval begin
    $M.$f(a::TrackedComplex, b::TrackedComplex)  = track($M.$f, a, b)
    $M.$f(a::TrackedComplex, b::Complex) = track($M.$f, a, b)
    $M.$f(a::Complex, b::TrackedComplex) = track($M.$f, a, b)
    function back(::typeof($M.$f), Δ::Complex, a::Complex, b::Complex)
      @back(a, Δ * $da)
      @back(b, Δ * $db)
    end
  end
end

# Eliminating ambiguity
import Base:^
^(a::TrackedComplex, b::Integer) = track(^, a, b)


import Flux: param
param(x::Complex) = TrackedComplex(complex(x))
x = 1.0
Tracked(Call(nothing), x, zero(x))
TrackedComplex(1.0+1im)
