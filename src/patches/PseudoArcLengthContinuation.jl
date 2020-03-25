using Zygote: @nograd
using Dates: now
@nograd now,string

import Base: +,iterate
using Base: RefValue
+(::Nothing, x::RefValue) = x
+(::Nothing, x::Float64) = x
+(::Nothing, x::Tuple) = x

iterate(::Nothing) = Nothing
