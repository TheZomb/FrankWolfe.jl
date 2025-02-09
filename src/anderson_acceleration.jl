using FrankWolfe
using LinearAlgebra
using Plots
using Random
using GLPK

# Anderson Acceleration Frank-Wolfe implementation
function anderson_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    initial_vertices; # special
    m=length(initial_vertices), # special    
    line_search::LineSearchMethod=Adaptive(),
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
    dual_gap_compute_frequency=1,
)
    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    function format_state(state)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    if m <= 0
        error("Parameter m must be positive for Anderson Acceleration.")
    end

    if length(initial_vertices) < m
        error("Number of initial vertices must be at least m.")
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    x = x0
    step_type = ST_ANDERSON
    time_start = time_ns()

    if (momentum !== nothing && line_search isa Union{Shortstep,Adaptive,Backtracking})
        @warn("Momentum-averaged gradients should usually be used with agnostic stepsize rules.",)
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end
    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    # instanciating container for gradient
    if gradient === nothing
        gradient = collect(x)
    end
    g_storage = (memory_mode == FrankWolfe.InplaceEmphasis() ? gradient : similar(gradient))

    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end


    function g(x::Vector{Float64})
        return grad!(g_storage, x) - x
    end
    X_k = [initial_vertices[i] - initial_vertices[i-1] for i in 2:m]
    G_k = [g(initial_vertices[i]) - g(initial_vertices[i-1]) for i in 2:m]
    # add zero vec at the starting
    push!(X_k, zeros(length(x)))
    push!(G_k, zeros(length(x)))
    circshift!(X_k, 1)
    circshift!(G_k, 1)

    function g(t::Int)
        return g(X_k[t])
    end

    while t <= max_iteration && norm(g(m)) >= max(epsilon, eps(float(eltype(x))))

        #####################
        # managing time and Ctrl-C
        #####################
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end
        #####################

        (Q, R) = qr(hcat(G_k...))

        # extend rows to R until size matches Q
        R = vcat(R, zeros(size(Q, 1) - size(R, 1), size(R, 2)))

        gamma = R \ (Q' * g(m))

        x_new = x + g(m) - hcat((X_k + G_k)...) * gamma
        g_new = grad!(g_storage, x_new) - x_new

        # update the set of relevant prev iterates
        X_k = circshift(X_k, -1)
        X_k[m] = copy(x_new - x)
        G_k = circshift(G_k, -1)
        G_k[m] = copy(g_new - G_k[m-1])

        x = x_new

        # go easy on runtime - only compute primal and dual if needed
        compute_iter = (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
        if compute_iter
            primal = f(x)
        end
        if t % dual_gap_compute_frequency == 0 || compute_iter
            dual_gap = 1.0
        end

        t += 1
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                tot_time,
                x,
                x,
                1.0,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                step_type,
            )
            if callback(state) === false
                break
            end
        end
    end

    return (x=x, v=x, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end