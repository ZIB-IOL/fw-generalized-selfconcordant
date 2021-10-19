using FrankWolfe
using LinearAlgebra
import FrankWolfe

struct SecondOrderSelfConcordant{H, HM} <: FrankWolfe.LineSearchMethod
    nu::Float64
    hessian::H
    hessian_storage::HM
    M::Float64
end

# to avoid printing the whole matrix
function Base.show(io::IO, mime::MIME"text/plain", ls::SecondOrderSelfConcordant)
    Base.show(io, mime, "SecondOrderSelfConcordant(nu=$(ls.nu), M=$(ls.M))")
end

function FrankWolfe.line_search_wrapper(
    line_search::SecondOrderSelfConcordant,
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
    return second_order_tvalue(line_search, x, d, dual_gap), L
end

function compute_second_order_delta(stepsizerule::SecondOrderSelfConcordant, x, d, e_x2)
    return if stepsizerule.nu == 2
        stepsizerule.M * norm(d)
    else
        (stepsizerule.nu - 2) / stepsizerule.nu * stepsizerule.M * norm(d)^(3 - stepsizerule.nu) * sqrt(e_x2)^(stepsizerule.nu - 2)
    end
end

function second_order_tvalue(stepsizerule::SecondOrderSelfConcordant, x, d, gap)
    stepsizerule.hessian(stepsizerule.hessian_storage, x)
    # local norm^2 of d = (x-v)
    e_x2 = dot(d, stepsizerule.hessian_storage, d)
    δ = compute_second_order_delta(stepsizerule, x, d, e_x2)
    if stepsizerule.nu == 2
        tv = inv(δ) * log(gap * δ / e_x2 + 1)
    elseif stepsizerule.nu == 3
        tv = gap / (δ * gap + e_x2)
    elseif stepsizerule.nu == 4
        tv = inv(δ) * (1 - exp(-δ * gap / e_x2))
    else
        error("Rule not written")
    end
    return min(1.0, tv)    
end

# Abstract linear map over matrices
# A HigherMatrix(Q) B == vec(A) Q vec(B)
struct HigherMatrix{T}
    hstorage::Matrix{T}
end

function LinearAlgebra.dot(x::AbstractMatrix, hm::HigherMatrix, y::AbstractMatrix)
    return dot(vec(x), hm.hstorage, vec(y))
end
