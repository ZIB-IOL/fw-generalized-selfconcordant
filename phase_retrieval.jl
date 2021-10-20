using LinearAlgebra
using MAT
using FrankWolfe
using JSON

include("monotonous_step.jl")
include("second_order_rule.jl")

phase_dir = filter!(endswith("mat"), readdir(joinpath(@__DIR__, "data/phase_retrieval/"), join=true))
push!(phase_dir, phase_dir[1])

phase_data = map(phase_dir) do f
    r = MAT.matread(f)
    (θ = r["theta"], W = r["W"], y = r["y"])
end

function build_objective_gradient(rt::NamedTuple)
    sy = sum(rt.y)
    ly = log.(rt.y)
    function f(x)
        sy + sum(1:size(rt.W, 2)) do i
            tp = dot(@view(rt.W[:,i]), x)
            tp * (log(tp) - ly[i]) - tp
        end
    end
    function ∇f(storage, x)
        storage .= 0
        for i in 1:size(rt.W, 2)
            w = @view(rt.W[:,i])
            tp = dot(w, x)
            @. storage += w * (log(tp) - ly[i])
        end
    end
    (f, ∇f)
end

function build_hessian(rt::NamedTuple)
    function hessian(storage, x)
        storage .= 0
        for i in 1:size(rt.W, 2)
            w = @view(rt.W[:,i])
            tp = dot(w, x)
            @. storage += w * w' / tp
        end
    end
    hessian
end

# basic tests
# r = MAT.matread(phase_dir[2])
# rt = (
#     θ = r["theta"],
#     W = r["W"],
#     y = r["y"],
# )
# (f, grad!) = build_objective_gradient(rt)
# hessian = build_hessian(rt)
# storage = rand(size(rt.W, 1))
# x = rand(size(rt.W, 1))

# using ReverseDiff
# grad!(storage, x)
# grad_ref = ReverseDiff.gradient(f, x)
# @show norm(grad_ref - storage)

# hstorage = Matrix{Float64}(undef, size(rt.W, 1), size(rt.W, 1))
# hessian(hstorage, x)
# href = ReverseDiff.hessian(f, x)
# @show norm(href - hstorage)

# roughly 90% quantile of Exp(1)
quantile_exp = 2.3

for (data_idx, rt) in enumerate(phase_data)
    (f, grad!) = build_objective_gradient(rt)
    hessian! = build_hessian(rt)
    n = size(rt.W, 1)
    hess_storage = Matrix{Float64}(undef, n, n)
    line_search_second = SecondOrderSelfConcordant(
        4.0, # nu,
        hessian!,
        hess_storage, # hessian storage
        1.0, # M
    )

    lmo = FrankWolfe.UnitSimplexOracle(50 * quantile_exp)
    # Note: -1 gradient x0 
    x0 = FrankWolfe.compute_extreme_point(lmo, -ones(n))
    storage = Vector{Float64}(undef, size(x0)...)

    (x, v, primal_so, dual_gap, traj_data_so) = FrankWolfe.frank_wolfe(
        f, grad!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=line_search_second,
        max_iteration=1000,
        gradient=storage,
    )

    (x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
        x -> f(x), grad!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        gradient=storage,
    )

    (x, v, primal_agnostic, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
        x -> f(x), grad!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=MonotonousStepSize(),
        max_iteration=5000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_backtracking) = FrankWolfe.frank_wolfe(
        x -> f(x), grad!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=5000,
        gradient=storage,
    )

    (xaw, v, primal_aw, dual_gap, traj_data_aw) = FrankWolfe.away_frank_wolfe(
        x -> f(x), grad!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=5000,
        gradient=storage,
    )

    open(joinpath(@__DIR__, "results", split(phase_dir[data_idx], "/")[end] * ".json"), "w") do f
        write(f,
            JSON.json((
                agnostic=traj_data_agnostic,
                backtracking=traj_data_backtracking,
                awaystep=traj_data_aw,
                monotonous=traj_data_monotonous,
                second_order=traj_data_so,
            )),
        )
    end
end
