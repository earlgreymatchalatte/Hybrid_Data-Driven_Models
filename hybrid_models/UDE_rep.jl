# Activate the project
using DrWatson
@quickactivate "NODE"

# Core packages
using Random, Dates, LinearAlgebra, Statistics

# File I/O
using CSV, DataFrames, DelimitedFiles, JLD2, JSON

# Argument parsing
using ArgParse

# Math and numerical tools
using NaNMath, ForwardDiff

# Plotting
using Plots

# Scientific computing
using DifferentialEquations, OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity

# Optimization
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Machine learning & autodiff
using Lux, Zygote, ComponentArrays, StableRNGs


"""
    parse_commandline()

Parse command-line arguments for job array index.

# Returns
- `Dict{String,Any}`: Parsed arguments, with at least the key `"arr"` indicating which job to run.
"""
function parse_commandline()
    args = ArgParseSettings()
    @add_arg_table args begin
        "--arr"
            help = "ID value of array"
            arg_type = Int
            default = 1
    end
    return parse_args(args)
end


"""
    step_mse(y_true, y_pred, t_true, s)

Compute a custom mean squared error between `y_true` and time-indexed values from `y_pred`, 
with a smoothness penalty weighted by `s`.

# Arguments
- `y_true`: Ground truth values (vector).
- `y_pred`: Model predictions (vector, assumed to be indexed beyond `t_true`).
- `t_true`: Time indices (vector of Ints).
- `s`: Smoothness weight (scalar).

# Returns
- A scalar value representing the custom MSE.
"""
function step_mse(y_true, y_pred, t_true, s)

    t_plus = t_true[2:end]
    t_minus = t_true[1:end-1]
    sol = sum(abs2.(y_true - [y_pred[i] for i in t_true .+1])) 
    + s * 0.3 * sum((abs2.(y_true[1:end-1]  - [y_pred[i] for i in t_plus .+1 ]) 
            + abs2.(y_true[2:end] - [y_pred[i] for i in t_minus .+1])))
    return sol / length(t_true)
end


"""
    get_U()

Constructs and initializes a Lux neural network with radial basis function activations for use in the UDE model.

# Returns
- `U`: Lux.Chain neural network model.
- `p_NN`: Initial parameters of the model.
- `_st`: Initial state of the model.
"""
function get_U()

    gr()
    # Set a random seed for reproducible behaviour
    rng = StableRNG(1)
    rbf(x) = exp.(-(x .^ 2))

    U = Lux.Chain(Lux.Scale(3), Lux.Dense(3,10, rbf), Lux.Dense(10,3, rbf),Lux.Dense(3,1)) # 

    # init params and state
    p_NN, st_NN  = Lux.setup(rng, U)
    _st = st_NN
    return U, p_NN, _st
end


"""
    friberg_drug_exp(du, u,p,t)

Defines the Friberg semi-mechanistic model of myelosuppression with drug effect [1].

This ODE system models the dynamics of proliferative and transit compartments in 
bone marrow under chemotherapy, where drug effect modulates cell proliferation.

# Arguments
- `du`: Output vector for derivatives.
- `u`: Current state vector `[log(prol), log(t1), log(t2), log(t3), log(circ)]`.
- `p`: Parameter tuple `(γ, MTT, c₀, slope, therapies, dosage)`.
- `t`: Current time (in hours).

# Returns
- Updates `du` in-place with `du/dt`.

# Notes
- The drug effect `E_drug` is active only within 24 hours of a therapy start time.
- `E_drug` is proportional to `slope * dosage` during this window.
"""
function friberg_drug_exp(du, u,p,t)
    # Unpack parameters
    γ, MTT, c₀, slope, therapies, dosage = p
    ktr = 4.0 / MTT

    # Exponentiate to get real values from log-transformed states
    prol, t1, t2, t3, circ = exp.(u)

    # Determine drug effect
    if t < minimum(therapies)
        E_drug = 0.0
    else
        # create local time from last therapy start,
        # where t_loc(start) = 0
        cond_th = therapies .<=t
        local_start = maximum(Int.(cond_th) .* therapies)
        local_dosage = dosage[findall(therapies .== local_start)]
        t_loc = t - local_start

        # on first treatment day, drug effect is assumed proportional to dose
        if t_loc <= 23
            E_drug = slope * local_dosage

        # on days without treatment, drug effect is asssumed zero
        else
            E_drug = [0.]
        end
    end

    # Differential equations (in log-space)
    du[1] = ktr * (1.0 - E_drug) * (c₀ / circ)^γ - ktr
    du[2] = ktr * prol / t1 - ktr
    du[3] = ktr * t1 / t2 - ktr
    du[4] = ktr * t2 / t3 - ktr
    du[5] = ktr * t3 / circ - ktr
end


"""
    ude_friberg!(du, u, p, t, p_true, U, _st)

Defines a universal differential equation (UDE) version [5] of the Friberg model [1] where the proliferative drive is entirely learned.

This version replaces the usual mechanistic proliferation term with a learned correction from the NN `U`.

# Arguments
- `du`: Output vector of derivatives (in-place).
- `u`: Current state vector (proliferation + 4 transit compartments).
- `p`: Parameter container with fields:
    - `:pmech` → `[MTT, c0]` scaled values.
    - `:pNN` → NN parameters.
- `t`: Time (hours).
- `p_true`: Friberg parameters including therapy schedule and dosage.
- `U`: Lux neural network.
- `_st`: Network state.

# Returns
- Updates `du` in-place with model derivatives.
"""
function ude_friberg!(du, u, p, t, p_true, U, _st)
    # friberg
    # unpack p
    _, MTT₀, c₀0, slope, therapies, dosage = p_true

    param_u = [195.0, 270.0]
    param_s = [195.0, 270.0]
    MTT, c₀ = p[:pmech] .* param_s .+ param_u
    p_U = p[:pNN]
    
    ktr = 4. / MTT
    # get vec components, really log(prol)
    prol, t1, t2, t3, circ = min.(max.(zeros(5), u), 1000. .* ones(5))

    # before first treatment, treatment has no effect
    if t < minimum(therapies)
        E_drug = [0.]
    else
        # create local time from last therapy start,
        # where t_loc(start) = 0
        cond_th = therapies .<=t
        local_start = maximum(Int.(cond_th) .* therapies)
        local_dosage = dosage[findall(therapies .== local_start)]
        t_loc = t - local_start

        # on first treatment day, drug effect is assumed proportional to dose
        if t_loc <= 23
            E_drug = slope * local_dosage

        # on days without treatment, drug effect is asssumed zero
        else
            E_drug = [0.]
        end
    end

    # UDE part
    u_input = Array([prol, circ, E_drug[1]])
    u_hat = U(u_input, p_U, _st)[1] # network prediction
    # UDE part ensuring non-negativity 
    ude_part = tanh(0.005*prol) * u_hat[1] * (1- E_drug[1])

    # Differential equations with UDE correction
    du[1] = ude_part - ktr * prol
    du[2] = ktr * prol- ktr * t1
    du[3] = ktr * t1 - ktr * t2
    du[4] = ktr * t2 - ktr * t3
    du[5] = ktr * t3 - ktr * circ
end


"""
    ude_friberg_preTL!(du, u, p, t, p_true, U, _st)

Friberg-based UDE model [1,5] where mechanistic parameters are fixed (`p_true`), and only the neural correction is active.

This is used for pretraining or transfer learning where the neural network `U` is trained while keeping
parameters like `MTT` and `c₀` fixed to known values.

# Arguments
- `du`: Derivatives vector (in-place).
- `u`: Current state vector.
- `p`: Parameter container with field `:pNN` (no `:pmech` needed).
- `t`: Current time.
- `p_true`: Ground-truth parameters `(γ, MTT, c₀, slope, therapies, dosage)`.
- `U`: Neural network (Lux.Chain).
- `_st`: NN state.

# Returns
- Updates `du` in-place.
"""
function ude_friberg_preTL!(du, u, p, t, p_true, U, _st)
    # friberg
    # unpack p
    gamma, MTT, c0, slope, therapies, dosage = p_true
    
    p_U = p[:pNN]
    
    ktr = 4. / MTT
    # get vec components, really log(prol)
    prol, t1, t2, t3, circ = min.(max.(zeros(5), u), 1000. .* ones(5))

    # before first treatment, treatment has no effect
    if t < minimum(therapies)
        E_drug = [0.]
    else
        # create local time from last therapy start,
        # where t_loc(start) = 0
        cond_th = therapies .<=t
        local_start = maximum(Int.(cond_th) .* therapies)
        local_dosage = dosage[findall(therapies .== local_start)]
        t_loc = t - local_start

        # on first treatment day, drug effect is assumed proportional to dose
        if t_loc <= 23
            E_drug = slope * local_dosage

        # on days without treatment, drug effect is asssumed zero
        else
            E_drug = [0.]
        end
    end
    u_input = Array([prol, circ, E_drug[1]])
        
    u_hat = U(u_input, p_U, _st)[1] # network prediction
    
    ude_part = tanh(0.005*prol) * u_hat[1] * (1- E_drug[1])

    # diff eq system, 
    du[1] = ude_part - ktr * prol
    du[2] = ktr * prol- ktr * t1
    du[3] = ktr * t1 - ktr * t2
    du[4] = ktr * t2 - ktr * t3
    du[5] = ktr * t3 - ktr * circ
end


"""
    objective(trial_config::Dict)

Run a full training and evaluation pipeline for a hybrid UDE-Friberg model using the specified trial configuration.

This function performs two stages of training:
1. **Pretraining**: Trains the neural correction using `ude_friberg_preTL!`, with fixed mechanistic parameters and Friberg-generated data.
2. **Fine-tuning**: Trains both the mechanistic parameters and neural network jointly using `ude_friberg!` on real patient data.

# Arguments
- `trial_config::Dict`: A dictionary of hyperparameters and metadata for the experiment.
    Required keys include:
    - `:pat`, `:cycle`, `:steps`, `:decay`, `:lr_start`, `:l2reg`, etc.
    - `:arr`: Trial index for organizing output.
    - `:s`, `:step_s`: Regularization weights for NN loss terms.
    - Other flags like `:step_mse_use`, `:skip_adam`, `:use_sophia`, etc.

# Effects
- Loads data from `../data_example/`.
- Saves:
    - Trained weights to `p_trained.jld2` and `p_trained_preTL.jld2`
    - Predictions to `nn_pred.csv`, `fri_pred.csv`
    - Trial metrics to `trial_result.json`
    - Full loss history to `losses.csv`

# Metrics Saved
- `:smse_NN`: NN performance on test data
- `:smse_friberg`, `:mse_friberg`: Friberg model baseline
- `:fu`, `:strange`: Long-term smoothness and plausibility
"""
function objective(trial_config) 

    @unpack pat, lr_start, l2reg, decay, decay_steps, steps, cycle, c, s, fu_start, step_s, step_opt, lr_startTL,
    l2regTL, decayTL, step_optTL, skip_adam, sigma, use_preTL_weights, fri_pen, step_mse_use, only_preTL, use_sophia,
    use_pop_weights, use_pop_params = trial_config

    ## === Setup and Data Loading === ##
    run_dir = "../results_example/"
    mkpath(run_dir)
    test_name ="pat_$(pat)_cyc$(cycle)_arr$(trial_config[:arr])" 
    test_dir = joinpath(run_dir, test_name)
    mkpath(test_dir)
    data_dir = "../data_example/"

    fri_dir = data_dir  # might be different for real data
    pats = DataFrame(CSV.File(joinpath(data_dir, "example_platelets.csv"); drop=[1]))
    dft = DataFrame(CSV.File(joinpath(data_dir, "example_treatment.csv"); drop=[1]))
    
    merge!(trial_config, Dict(:date => rightnow))
    merge!(trial_config, Dict(:mse_NN => Inf, :mse_friberg => Inf, :smse_NN => Inf, :smse_friberg => Inf))
    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end

    ## === Extract Patient Data === ##
    equals_pat(ID::Int64) = ID == pat
    t_data_true = filter(:ID => equals_pat, pats)."TIME"
    y_data_true = filter(:ID => equals_pat, pats)."Y"

    t_data = pats[(pats.ID .== pat) .&& (pats.CYCLE .<= cycle),:]."TIME"
    y_data = pats[(pats.ID .== pat) .&& (pats.CYCLE .<= cycle),:]."Y"


    t_data_test = pats[(pats.ID .== pat) .&& (pats.CYCLE .> cycle),:]."TIME"
    y_data_test = pats[(pats.ID .== pat) .&& (pats.CYCLE .> cycle),:]."Y"

    if isempty(t_data_test)
        trial_config[:empty] = true
        t_data_test = t_data_true
        y_data_test = y_data_true
    else
        trial_config[:empty] = false
    end

    t_len = last(t_data_true)

    therapies = filter(:ID => equals_pat, dft)."TIME" * 24.
    dosage = filter(:ID => equals_pat, dft)."DOSE_ST"

   ## === Load Precomputed Friberg Fit === ##
    fri_pat_dir = joinpath(fri_dir, "pat_$(pat)_cycle_$(cycle)")
    gamma = 0.316
    MTT = 195
    c0 =  270.
    slope = 2.
    fp = gamma, MTT, c0,  slope, therapies, dosage

    y0 = log.(c0 * ones(5))

    prob_trueode = ODEProblem(friberg_drug_exp, y0, (0., Float64.(t_data[end] * 24)), fp)


    fri_ppy = float(open(readdlm, joinpath(fri_pat_dir, "pc_fri.txt")))
    fri_p = [fri_ppy[1], fri_ppy[2], fri_ppy[4], fri_ppy[3]]
    

    fri_dict = Dict(:gamma => fri_p[1], :MTT => fri_p[2], :c0 => fri_p[3], :slope => fri_p[4])

    merge!(trial_config, fri_dict)
    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end

    # population parameters
    param_u = [195., 270.] 
    param_s = [195., 270.]


    ## === Baseline Friberg Simulation === ##
    fyo = fri_p[3] * ones(5)
    fp = fri_p[1], fri_p[2], fri_p[3], fri_p[4], therapies, dosage
    fprob = remake(prob_trueode, u0=log.(fyo), tspan=(0., t_data_true[end] * 24.), p=fp)
    fsol = solve(fprob, AutoTsit5(Rodas5P()),reltol=1e-10,saveat=24)
    t = fsol.t

    trial_config[:smse_friberg] = step_mse(NaNMath.log10.(y_data_test), NaNMath.log10.(exp.(fsol[5, :])), t_data_test, step_s)
    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end

    ## === Neural Network UDE Setup === ## 
    U, p_NN, _st = get_U()
    p_start_original = load_object(joinpath(data_dir, "p_trained_pop.jld2"))


    p_start = ComponentArray{Float64}(pmech=(([fri_p[2], fri_p[3]] .- param_u) ./ param_s), pNN= p_start_original)

    nn_friberg!(du, u, p, t) = ude_friberg!(du, u, p, t, fp, U, _st)
    nn_friberg_preTL!(du, u, p, t) = ude_friberg_preTL!(du, u, p, t, fp, U, _st)
    uy0 = fri_p[3] * ones(5)

    prob_nn = ODEProblem(nn_friberg!, uy0, (0., t_len * 24.), p_NN)
    prob_nn_TL = ODEProblem(nn_friberg_preTL!, uy0, (0., t_len * 24.), p_NN)

    ## === Predict Functions === ##
    function predict(theta, T=t)
        mech_p = theta[:pmech]  .* param_s .+ param_u 
        ny0 = mech_p[2] * ones(5)
        #println(mech_p[3])
        _prob = remake(prob_nn, u0=ny0, tspan = (T[1], T[end]), p = theta)
        Array(solve(_prob, AutoTsit5(Rodas5P(autodiff=false)), saveat = T, abstol=1e-9, reltol=1e-6,
                    dtmin = 0.05, tstops=therapies, force_dtmin = true,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end

    function predict_preTL(theta, T=t)
        #mech_p = theta[:pmech]  .* param_s .+ param_u 
        #println(mech_p[3])
        _prob = remake(prob_nn_TL, u0=uy0, tspan = (T[1], T[end]), p = theta)
        Array(solve(_prob, AutoTsit5(Rodas5P(autodiff=false)), saveat = T, abstol=1e-9, reltol=1e-6,
                    dtmin = 0.05, tstops=therapies, force_dtmin = true,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end

    ## === Loss Functions === ##
    # Steady state constraint
    function steadystate(theta, co = c)
        # trained friberg system gives zero if steady state is supplied, this should be conserved by NN 
        # by driving U(friberg_c0) to zero
        mech_p = theta[:pmech]  .* param_s .+ param_u 
        u_input = Array([mech_p[2], mech_p[2], 0.])
            
        u_hat = U(u_input, theta[:pNN], _st)[1] # network prediction
        
        ude_part = tanh(0.005*mech_p[2]) * u_hat[1]
        stst = ude_part - 4. / mech_p[1] * mech_p[2]
        return  co .* (sum(abs2.(stst)))
    end

    
    # Population prior penalty
    function pop_loss(pmech, sigma = 5.)
        # pop-params
        #println("pop loss pmech $(pmech)")
        gamma = 0.316
        MTT = 195
        c0 =  270.
        slope = 2.
        pop_p = Array([MTT, c0]) #, slope])
        l_pop = sum((log.(max.(1e-6, pmech)) - log.(pop_p)).^2 ./ (sigma ^ 2.))
        return l_pop
    end

    fri_sol_test = NaNMath.log10.(exp.(fsol[5,i] for i in t_data_test .+1))

    # Loss for pre-training
    function NNloss_preTL(theta)
        nn_sol = max.(1e-12, predict_preTL(theta, range(0, 350, step=1) .* 24)[5,:])
        #mech_p = theta[:pmech] .* param_s .+ param_u
        # only NN-params should be penalized by l2
        l2 = l2reg * sum(abs2, theta[:pNN][1:end-1])
        #stst = steadystate(theta)
        # late follow up should return to stable state
        fu_pred = nn_sol[fu_start:end]
        fu = sum(abs, diff(fu_pred))
        strange = sum(abs, (max.(1e-6, fri_p[3])) .- fu_pred)
        l_data  = mean(abs2, nn_sol[1:length(fsol[5,:])] .- exp.(fsol[5,:]))
        #l_pop = pop_loss(mech_p)
        sum_arr = Array{Float64}([l_data, l2, s * fu, s * strange ]) #, 10000 * abs(mech_p[1] - fri_p[2]), 10000 * abs(mech_p[2] - fri_p[3])])
        #println(sum_arr)
        res = sum(sum_arr)
        return res
    end

    ## === Pre-training with fixed Friberg parameters === ## 
    max_l = 50
 
    preTL_weights = p_start

    p_exp_best = [preTL_weights]
    
    p_exp_best = [p_start]
    best_loss = [Inf]
    losses = Float64[Inf]

    callback_fri = function (p, l)
        p_local = p.u
        st = p.original
        iter = p.iter
        #println("$(iter), $(l)")
        last_bestloss = best_loss[end]
        if l < best_loss[end]
            push!(best_loss, l)
            push!(p_exp_best, p_local)
        end
        
        push!(losses, l)
        
        if iter % decay_steps == 0
            if !step_opt
                current_lr = lr_start * exp(decay * floor(iter / decay_steps))
                Optimisers.adjust!(st, eta = current_lr)
                delta_l = sum(losses[end - Int(decay_steps-1) - 1: end - 1] - losses[end- Int(decay_steps-1) : end]) * 1/decay_steps 
                println("delta l = $(delta_l)")
                if delta_l < 0.01* l
                    return true
                end
            end
        end
        if iter > 20 
            return l - last_bestloss > 0.5 *  l
        else
            return false
        end
    end

    callbackLBGFS_friTL = function (p, l)
        p_local = p.u
        if l < best_loss[end]
            push!(best_loss, l)
            push!(p_exp_best, p_local)
        end
        
        push!(losses, l)
        return l - best_loss[end] > 0.5 *  l
    end

    adtype = Optimization.AutoZygote()
    optfTL = Optimization.OptimizationFunction((x, p_start) -> NNloss_preTL(x), adtype)

    if step_opt
        optprob_expTL = Optimization.OptimizationProblem(optfTL, ComponentVector{Float64}(p_start))
        res_exp = Optimization.solve(optprob_expTL, OptimizationOptimisers.Adam(lr_start), 
        callback = callback_fri, maxiters = 10)
        maxit = steps
        maxiter = decay_steps
        for i = 0:maxit
            last_bestloss = best_loss[end]
            lr = lr_start * exp(decay * i)
            optprob_exp = Optimization.OptimizationProblem(optfTL, p_exp_best[end])
            res_exp = Optimization.solve(optprob_exp, OptimizationOptimisers.Adam(lr), callback = callback_fri, maxiters = maxiter)
            last_bestloss = best_loss[end]

        end
    else
        i = 0
        optprob_expTL = Optimization.OptimizationProblem(optfTL, ComponentVector{Float64}(p_start))
        res_exp = Optimization.solve(optprob_expTL, OptimizationOptimisers.Adam(lr_start), 
        callback = callback_fri, maxiters = steps * decay_steps)
    end

    println("Adam preTL done")

    optprob_friL = Optimization.OptimizationProblem(optfTL, p_exp_best[end])
    
    try
        res_expL = Optimization.solve(optprob_friL, LBFGS(), callback = callbackLBGFS_friTL, maxiters = max_l)
    catch err
        println(err)
        println("Instability in TL lbfgs")
    end

    save_object(joinpath(test_dir, "p_trained_preTL.jld2"), p_exp_best[end])

    sol = predict(p_exp_best[end], range(0, 350, step=1) .* 24)[5,:]

    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end
    
    
    ## === Loss Function real patient training === ##
    function NNloss(theta, data_func)
        nn_sol = max.(1e-12, predict(theta, range(0, 350, step=1) .* 24)[5,:])
        mech_p = theta[:pmech] .* param_s .+ param_u
        # only NN-params should be penalized by l2
        l2 = l2regTL * sum(abs2, theta[:pNN][1:end-1])
        stst = steadystate(theta)
        # late follow up should return to stable state
        fu_pred = nn_sol[fu_start:end]
        fu = sum(abs, diff(fu_pred))
        strange = sum(abs, (log10.(max.(1e-6, mech_p[2])) .- NaNMath.log10.(fu_pred)))
        #l_data  = mean(abs2, NaNMath.log10.([nn_sol[i] for i in t_data .+1]) .- NaNMath.log10.(y_data))
        l_data = data_func(NaNMath.log10.(y_data), NaNMath.log10.(nn_sol), t_data)
        l_pop = pop_loss(mech_p)
        #friberg_penalty = 1/log10.(length(t_data)) * mean(abs2, (NaNMath.log10.([nn_sol[i] for i in t_data_test .+1]) .- fri_sol_test))
        friberg_penalty = 1/(length(t_data)) * mean(abs2, (NaNMath.log10.([nn_sol[i] for i in t_data_test .+1]) .- fri_sol_test))
        sum_arr = Array{Float64}([l_data, l2, s * stst, l_pop, s * fu, s * strange])#, friberg_penalty])
        #println(sum_arr)
        res = sum(sum_arr) + friberg_penalty
        #println("res $(res), friberg $(friberg_penalty)")
        return res
    end

    ## === Training === ##
    # First phase: Adam with step decay
    callback_exp = function (p, l)
        p_local = p.u
        st = p.original
        iter = p.iter
        #println("$(iter), $(l)")
        last_bestloss = best_exp_loss[end]
        if l < best_exp_loss[end]
            push!(best_exp_loss, l)
            push!(p_exp_best, p_local)
        end
        
        push!(losses_exp, l)
        
        if iter % decay_steps == 0
            if !step_optTL
                current_lr = lr_startTL * exp(decayTL * floor(iter / decay_steps))
                Optimisers.adjust!(st, eta = current_lr)
            end
            sol = predict(p_exp_best[end], range(0, 350, step=1) .* 24)[5,:]
            trial_config[:mse_NN] = sum(abs2, NaNMath.log10.([sol[i] for i in t_data_true .+1]) .- NaNMath.log10.(y_data_true))
            trial_config[:smse2_NN] = step_mse2(NaNMath.log10.(y_data_test), NaNMath.log10.(sol), t_data_test, step_s)
            open(joinpath(test_dir, "trial_result.json"), "w") do file
                JSON.print(file, trial_config)
            end
        end
        
        return l - last_bestloss > 0.5 *  l
    end

    best_exp_loss = [Inf]
    losses_exp = Float64[Inf]
    adtype = Optimization.AutoZygote()

    if step_mse_use
        println("step-mse-use")
        sstep_mse(y_true, y_pred, t_true, s=step_s) = step_mse2(y_true, y_pred, t_true, s)
        optf = Optimization.OptimizationFunction((x, p_start) -> NNloss(x, sstep_mse), adtype)
    else
        optf = Optimization.OptimizationFunction((x, p_start) -> NNloss(x, mse), adtype)
    end

    optprob_exp1 = Optimization.OptimizationProblem(optf, p_exp_best[end])

    if !skip_adam
        if step_optTL
            res_exp = Optimization.solve(optprob_exp1, OptimizationOptimisers.Adam(lr_startTL), 
            callback = callback_exp, maxiters = 10)
            maxit = steps
            maxiter = decay_steps
            for i = 0:maxit
                lr = lr_startTL * exp(decayTL * i)
                optprob_exp = Optimization.OptimizationProblem(optf, p_exp_best[end])
                res_exp = Optimization.solve(optprob_exp, OptimizationOptimisers.Adam(lr), callback = callback_exp, maxiters = maxiter)
            end
        else
            i=0
            res_exp = Optimization.solve(optprob_exp1, OptimizationOptimisers.Adam(lr_startTL),
            callback = callback_exp, maxiters = steps * decay_steps)
        end
    else

    end

    p_best_expL = [p_exp_best[end]]
    best_exp_loss = [minimum(losses_exp)]

    # LBFGS training
    losses_exp_L = Float64[minimum(losses_exp)]
    callbackSophia_exp = function (p, l)
        p_local = p.u
        if l < best_exp_loss[end]
            push!(best_exp_loss, l)
            push!(p_best_expL, p_local)
        end
        
        push!(losses_exp_L, l)

        if length(losses_exp_L) > 450
            delta_l = sum(losses_exp_L[end - Int(50-1) - 1: end - 1] - losses_exp_L[end- Int(50-1) : end]) * 1/50
            if delta_l < 0.001 * l
                return true
            end
        end
        return l - best_exp_loss[end] > 0.5 *  l
    end

    losses_exp_L2 = Float64[minimum(losses_exp_L)]

    callbackLBGFS_exp = function (p, l)
        p_local = p.u
        if l < best_exp_loss[end]
            
            push!(best_exp_loss, l)
            push!(p_best_expL, p_local)
        end
        
        push!(losses_exp_L2, l)

        if length(losses_exp_L2) > 25
            delta_l = sum(losses_exp_L[end - Int(25-1) - 1: end - 1] - losses_exp_L[end- Int(25-1) : end]) * 1/25
            if delta_l < 0.001 * l
                return true
            end
        end
        return l - best_exp_loss[end] > 0.5 *  l
    end

    # adapted Loss function for SOPHIA and LBFGS finetuning
    function NNlossl(theta, data_func)
        nn_sol = max.(1e-12, predict(theta, range(0, 350, step=1) .* 24)[5,:])
        mech_p = theta[:pmech] .* param_s .+ param_u
        # only NN-params should be penalized by l2
        l2 = l2regTL * sum(abs2, theta[:pNN][1:end-1])
        stst = steadystate(theta)
        # late follow up should return to stable state
        fu_pred = nn_sol[fu_start:end]
        l_data = data_func(NaNMath.log10.(y_data), NaNMath.log10.(nn_sol), t_data)
        #mean(abs2, NaNMath.log10.([nn_sol[i] for i in t_data .+1]) .- NaNMath.log10.(y_data))
        l_pop = pop_loss(theta[:pmech].* param_s .+ param_u, sigma)
        fu = sum(abs, diff(fu_pred))
        strange = sum(abs, (log10.(max.(1e-6, mech_p[2])) .- NaNMath.log10.(fu_pred)))
        friberg_penalty = fri_pen * 1/(length(t_data)) * mean(abs2, (NaNMath.log10.([nn_sol[i] for i in t_data_test .+1]) .- fri_sol_test))
        #l_pop = 0.
        return l_data + l2 + l_pop + s * (stst + fu + strange) + friberg_penalty
    end



    adtype2 = AutoZygote()
    if step_mse_use
        println("step-mse-use")
        sstep_mse(y_true, y_pred, t_true, s=step_s) = step_mse2(y_true, y_pred, t_true, s)
        optf_L = Optimization.OptimizationFunction((x, p_NN) -> NNlossl(x, sstep_mse), adtype2)
    else
        optf_L = Optimization.OptimizationFunction((x, p_NN) -> NNlossl(x, mse), adtype2)
    end
    optprob_L = Optimization.OptimizationProblem(optf_L, p_best_expL[end])

    if use_sophia
        try
            res_expL = Optimization.solve(optprob_L, Optimization.Sophia(), callback = callbackSophia_exp, maxiter=450)
        catch err
            println(err)
            println("Instability in sophia")
        end

        len_sophia = length(losses_exp_L)
    else
        len_sophia = 0
    end

    try
        optprob_L2 = Optimization.OptimizationProblem(optf_L, p_best_expL[end])

        res_expL = Optimization.solve(optprob_L2, LBFGS(), callback = callbackLBGFS_exp, maxiters = 50)
    catch err
        println(err)
        println("Instability in lbfgs")
    end


    println("Final training loss after $(length(losses_exp_L)) iterations: $(losses_exp_L[end])")

    ## === Final Predictions and Save === ##
    # Rename the best candidate
    p_exp_LBGFS = p_best_expL[end]

    nnsol_exp_L = predict(p_exp_LBGFS, range(0, 350, step=1) .* 24)[5,:]

    # step-mse
    smse_NN = step_mse(NaNMath.log10.(y_data_test), NaNMath.log10.(nnsol_exp_L), t_data_test, step_s)
    smse_friberg = step_mse(NaNMath.log10.(y_data_test), NaNMath.log10.(exp.(fsol[5,:])), t_data_test, step_s)

    fu_pred = nnsol_exp_L[fu_start:end]
    trial_config[:smse_NN] = smse_NN
    trial_config[:smse_friberg] = smse_friberg
    trial_config[:fu] = sum(abs, diff(fu_pred)) 
    trial_config[:strange] = sum(abs, (log10.(max.(1e-6, p_exp_LBGFS[:pmech][2])) .- NaNMath.log10.(fu_pred)))

    save_object(joinpath(test_dir, "p_trained.jld2"), p_exp_LBGFS)
    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end

    writedlm(joinpath(test_dir, "nn_pred.csv"), nnsol_exp_L[1:length(fsol[5,:])], ',')
    writedlm(joinpath(test_dir, "fri_pred.csv"), exp.(fsol[5,:]), ',')
    writedlm(joinpath(test_dir, "losses.csv"),vcat(vcat(losses_exp, losses_exp_L), losses_exp_L2), ',')
    
end


"""
    main()

Main entry point for running a training trial for the hybrid UDE-Friberg model.

This function:
- Parses a job index from the command line (`--arr`) to select one configuration from a hyperparameter grid.
- Loads patient and treatment data.
- Constructs a `trial_config` dictionary containing all hyperparameters and metadata for the trial.
- Calls `objective(trial_config)` to run the experiment.
- Logs the total runtime.

# Required
- `example_platelets.csv` in `../data_example/`
- Command-line argument `--arr` to select a specific configuration

# Example
```bash
julia script.jl --arr=3
"""
function main()
    t1 = time()
    println("running")

    parsed_args = parse_commandline()
    arr = parsed_args["arr"]

     # load data
     data_dir = "../data_example/"
     pats = DataFrame(CSV.File(joinpath(data_dir, "example_platelets.csv"); drop=[1]))
     data_true = pats."ID"


# === Hyperparameters === #
    lr_start = 0.005
    l2reg = 10
    decay = -0.5
    couple = 0.1
    decay_steps = 600
    steps = 9
    pat = data_true
    s = 0.1
    fu_start = 250
    step_s = 1.
    step_opt = true

    fri_pen = 1.
    sigma = 5.
    step_mse_use = true
    only_preTL = false
    step_optTL = true


    # Fine-tuning phase options
    lr_startTL = [0.005]
    l2regTL = [0.0]
    decayTL = [-0.3]
    skip_adam = [false]
    use_sophia = [true]
    use_preTL_weights = true
    use_pop_weights = false
    use_pop_params = [false]


    cycle = [1, 2, 3, 4, 5, 6]

    indices = CartesianIndices((length(cycle), length(pat), length(lr_startTL), length(l2regTL), length(decayTL), length(skip_adam), length(use_sophia), length(use_pop_params)))

    trial = Dict{Symbol,Any}(
        :lr_start=> lr_start,
        :l2reg=> l2reg,
        :decay => decay,
        :decay_steps => decay_steps,
        :steps => steps,
        :cycle => cycle[indices[arr].I[1]],
        :pat=> pat[indices[arr].I[2]],
        :c => couple,
        :s => s,
        :arr => arr,
        :fu_start => fu_start,
        :step_s => step_s,
        :step_opt => step_opt,
        :lr_startTL => lr_startTL[indices[arr].I[3]],
        :l2regTL => l2regTL[indices[arr].I[4]],
        :decayTL => decayTL[indices[arr].I[5]],
        :step_optTL => step_optTL,
        :skip_adam => skip_adam[indices[arr].I[6]],
        :sigma => sigma,
        :use_preTL_weights => use_preTL_weights,
        :fri_pen => fri_pen,
        :step_mse_use => step_mse_use,
        :only_preTL => only_preTL,
        :use_sophia => use_sophia[indices[arr].I[7]],
        :use_pop_weights => use_pop_weights,
        :use_pop_params => use_pop_params[indices[arr].I[8]])

    # === Run trial === #
    objective(trial)

    elapsed_time = time() - t1
    println("Elapsed time: ", elapsed_time, " seconds")
    println("finished")
end

main()