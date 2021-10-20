using LinearAlgebra
using FrankWolfe
using CSV
using DataFrames
using ReverseDiff
using JSON

include(joinpath(@__DIR__, "second_order_rule.jl"))
include(joinpath(@__DIR__, "monotonous_step.jl"))

log_dir = filter!(endswith("csv"), readdir(joinpath(@__DIR__, "data/logreg/"), join=true))
push!(log_dir, log_dir[1])

# build feature and outcome vectors
function preprocess_dataframe(df)
    index = df.target
    target_correct_scale = sort!(unique(df.target)) == [-1,1]
    if !target_correct_scale
        @assert(sort!(unique(df.target)) == [1,2])
    end
    nf = size(df, 2) - 3
    a_s = Vector{NTuple{nf, Float64}}()
    ys = Vector{Float64}()
    sizehint!(a_s, size(df, 1))
    sizehint!(ys, size(df, 1))
    for r in eachrow(df)
        if target_correct_scale
            push!(ys, r.target)
        else
            push!(ys, r.target * 2 - 3)
        end
        push!(a_s, values(r[4:end]))
    end
    return (a_s, ys)
end

function build_objective_gradient(df, mu)
    (a_s, ys) = preprocess_dataframe(df)
    # just flexing with unicode
    # reusing notation from Bach 2010 Self-concordant analysis for LogReg
    ℓ(u) = log(exp(u/2) + exp(-u/2))
    dℓ(u) = -1/2 + inv(1 + exp(-u))
    n = length(ys)
    invn = inv(n)
    function f(x)
        err_term = invn * sum(eachindex(ys)) do i
            dtemp = dot(a_s[i], x)
            ℓ(dtemp) - ys[i] * dtemp / 2
        end
        pen_term = mu * dot(x, x) / 2
        err_term + pen_term
    end
    function grad!(storage, x)
        storage .= 0
        for i in eachindex(ys)
            dtemp = dot(a_s[i], x)
            @. storage += invn * a_s[i] * (dℓ(dtemp) - ys[i] / 2)
        end
        @. storage += mu * x
        storage
    end
    (f, grad!)
end

function build_hessian(df, mu)
    (a_s, ys) = preprocess_dataframe(df)
    n = length(collect(ys))
    invn = inv(n)
    function hessian(storage, x)
        storage .= 0
        for i in eachindex(ys)
            dtemp = dot(a_s[i], x)
            array = collect(a_s[i])
            out_prod = array * array' 
            @. storage += invn * out_prod/((1 + exp(ys[i] * dtemp))*(1 + exp(-ys[i] * dtemp)))
        end
        storage .= storage + mu * I
    end
end

function compute_self_concordance_paremeter(df)
    (a_s, ys) = preprocess_dataframe(df)
    parameter = 0
    for i in eachindex(ys)
        norm_val = norm(a_s[i])
        if(norm_val > parameter)
            parameter = norm_val
        end
    end
    parameter
end

for df_idx in 1:length(log_dir)-1
    df = CSV.read(log_dir[df_idx], DataFrame)
    (f0, grad0!) = build_objective_gradient(df,  1/sqrt(size(df, 1)))

    #Build the Hessian computation
    hessian = build_hessian(df, 1/sqrt(size(df, 1)))
    M = compute_self_concordance_paremeter(df)
    hess_storage = Matrix{Float64}(undef, length(f0.a_s[1]), length(f0.a_s[1]))
    line_search_second = SecondOrderSelfConcordant(
        2.0, # nu,
        hessian,
        hess_storage,
        M, # M
    )

    # similar to Frank-Wolfe Newton parameters
    lmo = FrankWolfe.LpNormLMO{1}(1)
    x0 = FrankWolfe.compute_extreme_point(lmo, -ones(length(f0.a_s[1])))
    storage = collect(x0)

    # warning: extremely slow
    (x, v, primal_so, dual_gap, traj_data_so) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=line_search_second,
        max_iteration=10000,
        gradient=storage,
    )

    """
    (x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=10000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_backtracking) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=10000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=MonotonousStepSize(),
        linesearch_tol=1e-8,
        max_iteration=10000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_aw) = FrankWolfe.away_frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=10000,
        gradient=storage,
    )
    """

    open(joinpath(@__DIR__, "results", split(log_dir[df_idx], "/")[end] * "_second_order.json"), "w") do f
        write(f, JSON.json((
            #agnostic=traj_data_agnostic,
            #backtracking=traj_data_backtracking,
            #awaystep=traj_data_aw,
            #monotonous=traj_data_monotonous,
            second_order=traj_data_so,
        )))
    end
end

# sanity check for gradient
# x = 1/20 * randn(size(df, 2) - 3)
# f0(x)
# storage = similar(x)
# grad0!(storage, x)
# @show norm(storage - ReverseDiff.gradient(f0, x))
