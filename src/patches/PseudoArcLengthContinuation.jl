using Zygote: @nograd
using Dates: now
@nograd now,string

import Base: +
+(::Nothing, x::Float64) = x
+(::Nothing, x::Tuple) = x
