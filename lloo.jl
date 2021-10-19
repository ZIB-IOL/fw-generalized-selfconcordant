using FrankWolfe
using LinearAlgebra
using SparseArrays

function lloo_probability_simplex(x, radius, c)
    n = length(x)
    @assert length(c) == n
    d = sqrt(n) * radius
    Δ = min(d/2, 1)
    imin = argmin(c)
    p_plus = FrankWolfe.ScaledHotVector(Δ, imin, n)
    p_minus = zero(x)
    sindices = sortperm(c, rev=true)
    k = 1
    while k < n && sum(x[i] for i in sindices[1:k]) < Δ
        k += 1
    end
    for j in 1:k-1
        p_minus[sindices[j]] = x[sindices[j]]
    end
    sum_or0 = if k > 1
        sum(p_minus[sindices[j]] for j in 1:k-1)
    else
        0
    end
    p_minus[sindices[k]] = Δ - sum_or0
    return x + p_plus - p_minus
end

function frank_wolfe_lloo(
    f,
    grad!,
    lmo,
    x0,
    hessian,
    hess_storage,
    sigma,
    M_f;
    max_iteration=10000,
    max_time=3600,
)
    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    #tt = regular
    traj_data = []
    time_start = time_ns()

    if !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = convert(Array{float(eltype(x))}, x)
        else
            x = convert(Array{eltype(x)}, x)
        end
    end
    # instanciating container for gradient
    d = similar(x)
    gradient = similar(x)
    
    #Compute information for first iteration
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    h_k = dot(x, gradient) - dot(v, gradient)
    r_k = sqrt(2*dual_gap/sigma)
    local_v = lloo_probability_simplex(x, r_k, gradient)
    #@emphasis(emphasis, d = local_v - x)
    @. d = local_v - x
    hessian(hess_storage, x)
    hess_mult = dot(d, hess_storage, d)
    e=sqrt(hess_mult)*M_f/2
    alpha_k = min(h_k*M_f^2 /(4*e^2),1)*(1/(1+e))
    h_k = h_k * exp(-alpha_k/2)
    r_k = r_k * sqrt(exp(-alpha_k /2))

    #@emphasis(emphasis, x = x + gamma * d)
    @. x = x + alpha_k * d
    while t <= max_iteration && (time_ns() - time_start) / 1e9 <= max_time
        grad!(gradient, x)

        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = dot(x, gradient) - dot(v, gradient)
        state = (
            t=t,
            primal=primal,
            dual=primal - dual_gap,
            dual_gap=dual_gap,
            time=(time_ns() - time_start) / 1e9,
        )

        push!(traj_data, state)
        print("\n ")
        print(t)
        print(" ")
        print(primal)
        print(" ")
        print(dual_gap)
        print(" ")
        print((time_ns() - time_start) / 1e9)
        print(" ")

        local_v = lloo_probability_simplex(x, r_k, gradient)
        #@emphasis(emphasis, d = local_v - x)
        @. d = local_v - x
        hessian(hess_storage, x)
        hess_mult = dot(d, hess_storage, d)
        e=sqrt(hess_mult)*M_f/2
        alpha_k = min(h_k*M_f^2 /(4*e^2),1)*(1/(1+e))
        h_k = h_k * exp(-alpha_k/2)
        r_k = r_k * sqrt(exp(-alpha_k /2))
        #@emphasis(emphasis, x = x + gamma * d)
        @.  x = x + alpha_k * d
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now! 
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = dot(x, gradient) - dot(v, gradient)
    return x, v, primal, dual_gap, traj_data
end




# sanity check
# x = rand(4)
# x ./= sum(x)
# c = randn(4)
# r = 0.6
# p = lloo_probability_simplex(x, r, c)
# norm(p, 1) == 1
# norm(p - x, 1) == sqrt(4) * r

# do not use.
# function lloo_polytope(radius, d, lmo)
#     Δ = min(1, 1) # TODO
#     n = length(c)
#     ls = [dot(vi, c) for vi in vs]
#     sindices = sortperm(ls, rev=true)
#     k = 1
#     while k < n && sum(λs[i] for i in sindices[1:k]) < Δ
#         k += 1
#     end
#     p_minus = spzeros(n)
#     for j in 1:k-1
#         @. p_minus += λs[sindices[j]] * vs[sindices[j]]
#     end
#     if k > 1
#         p_minus .+= (Δ - sum(λs[sindices[1:k-1]])) * vs[sindices[k]]
#     end
#     v_lmo = FrankWolfe.compute_extreme_point(lmo, c)
#     p_plus = Δ * v_lmo
#     return sum(λs .* vs) + p_plus - p_minus
# end

# computing the strong convexity parameter σ_f for portfolio optimization
function compute_convexity_parameter(W)
    (n, p) = size(W)
    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    denominator_overestimator = map(1:p) do t
        w = @view(W[:,t])
        vmax = FrankWolfe.compute_extreme_point(lmo, -w)
        vmin = FrankWolfe.compute_extreme_point(lmo, w)
        b = max(dot(vmax, w), -dot(vmin, w))
        @assert(b >= 0)
        b
    end
    hessian_under = zeros(n, n)
    for t in 1:p
        w = @view(W[:,t])
        hessian_under .+= w * w' / denominator_overestimator[t]^2
    end
    return minimum(LinearAlgebra.eigvals(hessian_under))
end

#compute_convexity_parameter(raw_data[1])
