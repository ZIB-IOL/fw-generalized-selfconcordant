using FrankWolfe

"""
    MonotonousStepSize{F}

Represents a monotonous open-loop step size.
Contains a halving factor increased at each iteration until there is primal progress
`gamma = 2 / (t + 2) * 2^(-N)`
"""
mutable struct MonotonousStepSize{F} <: FrankWolfe.LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonousStepSize(f::F) where {F <: Function} = MonotonousStepSize{F}(f, 0)
MonotonousStepSize() = MonotonousStepSize(x -> true)

function FrankWolfe.line_search_wrapper(
    line_search::MonotonousStepSize,
    t,
    f,
    grad!,
    x,
    d,
    gradient,
    dual_gap,
    L,
    gamma0,
    linesearch_tol,
    step_lim,
    gamma_max,
)
    gamma = 2.0^(1-line_search.factor) / (2 + t)
    xnew = x - gamma * d
    f0 = f(x)
    while !line_search.domain_oracle(xnew) || f(xnew) > f0
        line_search.factor += 1
        gamma = 2.0^(1-line_search.factor) / (2 + t)
        @. xnew = x - gamma * d
    end
    return gamma, L
end
