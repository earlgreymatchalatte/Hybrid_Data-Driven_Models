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
    get_U(config::Int)

Constructs a Lux neural network `U` with different architectures based on the `config` parameter.
Initializes the parameters and state with a fixed RNG seed for reproducibility.

# Arguments
- `config::Int`: Determines the network architecture (0 or 1).

# Returns
- `U`: The Lux.Chain model
- `p_NN`: Initialized model parameters
- `_st`: Initialized model state
"""
function get_U(config)

    gr()
    # Set a random seed for reproducible behaviour
    rng = StableRNG(1)
    rbf(x) = exp.(-(x .^ 2))

    if config == 0
        U = Lux.Chain(Lux.Scale(3), Lux.Dense(3, 10, rbf), Lux.Dense(10, 3, rbf),Lux.Dense(3,1; init_weight=WeightInitializers.zeros32, init_bias=WeightInitializers.zeros32))
    elseif config == 1
        U = Lux.Chain(Lux.Scale(3), Lux.Dense(3, 3, rbf), Lux.Dense(3,1; ; init_weight=WeightInitializers.zeros32, init_bias=WeightInitializers.zeros32))
    end
    
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

Defines a neural-augmented version of the Friberg model (UDE) [1,5], where a neural network 
`U` learns a correction to the first differential equation.

# Arguments
- `du`: Derivative vector (in-place update).
- `u`: State vector (length 5): `[log(prol), log(t1), log(t2), log(t3), log(circ)]`.
- `p`: Parameter container with fields:
    - `pmech`: parameters for mechanistic part (vector of 3 values).
    - `pNN`: neural network parameters.
- `t`: Time (hours).
- `p_true`: Friberg parameters including therapy schedule.
- `U`: Neural network model (Lux.Chain).
- `_st`: Neural network state.

# Returns
- Updates `du` in-place.
"""
function ude_friberg!(du, u, p, t, p_true, U, _st)
    # Unpack true parameters for context
    γ₀, MTT₀, c₀, slope, therapies, dosage = p_true

    # Rescaling parameters: affine transformation
    param_u = [0.316, 195.0, 270.0]
    param_s = [0.316, 195.0, 270.0]

    γ, MTT, c₀ = p[:pmech] .* param_s .+ param_u
    p_U = p[:pNN]

    ktr = 4.0 / MTT

    # get vec components, really log(prol), limit to 0, 1000] for stability
    prol, t1, t2, t3, circ = min.(max.(zeros(5), u), 1000. .* ones(5))

    # Drug effect computation
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

    # UDE correction
    u_input = Array([prol, circ, E_drug[1]])
    u_hat = U(u_input, p_U, _st)[1] # network prediction
    
    # ensure non-negativity 
    ude_part = tanh(0.005*prol) * u_hat[1]

    # Differential equations with UDE correction
    du[1] = ktr * prol * ((1.0 - E_drug) * (c₀ / circ)^γ - 1.0) + ude_part
    du[2] = ktr * prol - ktr * t1
    du[3] = ktr * t1 - ktr * t2
    du[4] = ktr * t2 - ktr * t3
    du[5] = ktr * t3 - ktr * circ
end


"""
    objective(trial_config::Dict)

Runs a full training and evaluation trial for the UDE-Friberg model using specified hyperparameters.

This function:
- Loads patient and treatment data
- Simulates the Friberg model as a baseline
- Constructs and trains a neural differential equation (UDE) model
- Evaluates model performance
- Saves results and predictions to disk

# Arguments
- `trial_config::Dict`: Dictionary of hyperparameters and identifiers for the trial (e.g., learning rate, cycle, patient ID, etc.).

# Outputs
- Saves trained model, predictions, and metrics to files in the corresponding result directory.
"""
function objective(trial_config::Dict)

    @unpack pat, lr_start, l2reg, decay, decay_steps, steps, cycle, U_config, s, step_s, fu_start = trial_config
    c = 1.


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
    

    # Log start time and initialize JSON
    merge!(trial_config, Dict(:date => rightnow))
    merge!(trial_config, Dict(:mse_NN => Inf, :mse_friberg => Inf))
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


    # if all data is used for training, still generate evaluation 
    trial_config[:empty] = isempty(t_data_test)
    if trial_config[:empty]
        t_data_test = t_data_true
        y_data_test = y_data_true
    end

    t_len = last(t_data_true)

    therapies = filter(:ID => equals_pat, dft)."TIME" * 24.
    dosage = filter(:ID => equals_pat, dft)."DOSE_ST"

    ## === Load Precomputed Friberg Fit === ##
    fri_pat_dir = joinpath(fri_dir, "pat_$(pat)_cycle_$(cycle)")
    # population paramterers
    gamma = 0.316
    MTT = 195
    c0 =  270.
    slope = 2.
    fp = gamma, MTT, c0,  slope, therapies, dosage
    y0 = log.(c0 * ones(5))
    prob_trueode = ODEProblem(friberg_drug_exp, y0, (0., Float64.(t_data[end] * 24)), fp)
    # individual parameters
    fri_ppy = float(open(readdlm, joinpath(fri_pat_dir, "pc_fri.txt")))
    fri_p = [fri_ppy[1], fri_ppy[2], fri_ppy[4], fri_ppy[3]]
    fri_dict = Dict(:gamma => fri_p[1], :MTT => fri_p[2], :c0 => fri_p[3], :slope => fri_p[4])
    merge!(trial_config, fri_dict)
    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end


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
    U, p_NN, _st = get_U(trial_config[:U_config])
    # scaling population parameters
    param_u = [0.316, 195., 270.] #, 2.]
    param_s = [0.316, 195., 270.] #, 2.]
    p_start = ComponentArray{Float64}(pmech=((fri_p[1:3] .- param_u) ./ param_s), pNN= p_NN)

    # setup UDE
    nn_friberg!(du, u, p, t) = ude_friberg!(du, u, p, t, fp, U, _st)
    uy0 = fri_p[3] * ones(5)
    prob_nn = ODEProblem(nn_friberg!, uy0, (0., t_len * 24.), p_NN)

    # Predict function
    function predict(theta, T=t)
        mech_p = theta[:pmech]  .* param_s .+ param_u 
        ny0 = mech_p[3] * ones(5)
        #println(mech_p[3])
        _prob = remake(prob_nn, u0=ny0, tspan = (T[1], T[end]), p = theta)
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
        u_input = Array([mech_p[3], mech_p[3], 0.])
            
        u_hat = U(u_input, theta[:pNN], _st)[1] # network prediction
        
        ude_part = tanh(0.005*mech_p[3]) * u_hat[1]
        stst = ude_part
        return  sum(abs2.(stst))
    end

    # Population prior penalty
    function pop_loss(pmech)
        # pop-params
        gamma = 0.316
        MTT = 195
        c0 =  270.
        slope = 2.
        pop_p = Array([gamma, MTT, c0]) #, slope])
        l_pop = sum((log.(max.(1e-6, pmech)) - log.(pop_p)).^2 ./ (5 ^ 2.))
        return l_pop
    end

    # Main loss
    function NNloss(theta)
        nn_sol = max.(1e-12, predict(theta, range(0, 350, step=1) .* 24)[5,:])
        mech_p = theta[:pmech] .* param_s .+ param_u
        # only NN-params should be penalized by l2
        l2 = l2reg * sum(abs2, theta[:pNN])
        stst = steadystate(theta)
        # late follow up should return to stable state
        fu_pred = nn_sol[fu_start:end]
        fu = sum(abs, diff(fu_pred))
        strange = sum(abs, (log10.(max.(1e-6, mech_p[3])) .- NaNMath.log10.(fu_pred)))
        l_data  = sum(abs2, NaNMath.log10.([nn_sol[i] for i in t_data .+1]) .- NaNMath.log10.(y_data))
        l_pop = pop_loss(mech_p)
        sum_arr = Array{Float64}([l_data, l2, s* stst, l_pop, s * fu, s * strange])
        res = sum(sum_arr)
        return res
    end

    ## === Training === ##

    
    # First phase: Adam with step decay
    p_exp_best = [p_start]
    best_exp_loss = [Inf]
    losses_exp = Float64[Inf]

    callback_exp = function (p, l)
        p_local = p.u
        st = p.original
        iter = p.iter
        last_bestloss = best_exp_loss[end]
        if l < best_exp_loss[end]
            push!(best_exp_loss, l)
            push!(p_exp_best, p_local)
        end
        push!(losses_exp, l)
        if iter % decay_steps == 0
            current_lr = lr_start * exp(decay * floor(iter / decay_steps))
            Optimisers.adjust!(st, eta = current_lr)
            sol = predict(p_local, range(0, 350, step=1) .* 24)[5,:]
            trial_config[:mse_NN] = sum(abs2, NaNMath.log10.([sol[i] for i in t_data_true .+1]) .- NaNMath.log10.(y_data_true))
            open(joinpath(test_dir, "trial_result.json"), "w") do file
                JSON.print(file, trial_config)
            end
        end
        
        # early stopping if big jumps in function
        return l - last_bestloss > 0.5 *  l
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p_start) -> NNloss(x), adtype)
    optprob_exp1 = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_start))
    try 
        res_exp = Optimization.solve(optprob_exp1, OptimizationOptimisers.Adam(lr_start),
        callback = callback_exp, maxiters = steps * decay_steps)
    catch err
        println(err)
        println("Instability in Adam")
    end

        
    ## === LBFGS Fine-tuning === ##
    p_best_expL = [p_exp_best[end]]
    best_exp_loss = [minimum(losses_exp)]
    losses_exp_L = Float64[minimum(losses_exp)]
    callbackLBGFS_exp = function (p, l)
        p_local = p.u
        if l < best_exp_loss[end]
            push!(best_exp_loss, l)
            push!(p_best_expL, p_local)
        end
        
        push!(losses_exp_L, l)
        return l > 1e20
    end

    adtype2 = AutoZygote()
    optf_L = Optimization.OptimizationFunction((x, p_NN) -> NNloss(x), adtype2)
    optprob_L = Optimization.OptimizationProblem(optf_L, p_best_expL[end])

    max_l = 1000

    try
        res_expL = Optimization.solve(optprob_L, LBFGS(), callback = callbackLBGFS_exp, maxiters = max_l)
    catch err
        println(err)
        println("Instability in lbfgs")
    end

    ## === Final Predictions and Save === ##
    p_exp_LBGFS = p_best_expL[end]
    nnsol_exp_L = predict(p_exp_LBGFS)
    smse_NN = step_mse(NaNMath.log10.(y_data_test), NaNMath.log10.(nnsol_exp_L[5, :]), t_data_test, step_s)
    trial_config[:smse_NN] = smse_NN

    save_object(joinpath(test_dir, "p_trained.jld2"), p_exp_LBGFS)
    open(joinpath(test_dir, "trial_result.json"), "w") do file
        JSON.print(file, trial_config)
    end
    
    writedlm(joinpath(test_dir, "nn_pred.csv"), nnsol_exp_L[5,:], ',')
    writedlm(joinpath(test_dir, "fri_pred.csv"), exp.(fsol[5,:]), ',')
end


"""
    main()

Main entry point for running a grid search trial on the UDE-Friberg model using predefined hyperparameter values
for all patients and data availability scenarios. 

Selects one trial from a Cartesian product of hyperparameters using a command-line argument `arr`,
then calls `objective(...)` to run training and evaluation.

# Requires
- `"arr"` from command-line arguments.
- Example data in `../data_example/`.

# Saves
- Predictions, model checkpoints, and trial results to `../results_example/`.
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
    

    # Grid search hyperparameters
    lr_start = [0.001]
    l2reg = [0.0]
    decay = [0.0]
    decay_steps = [600]
    steps = [9]
    cycle = [1, 2, 3, 4, 5, 6]
    pat = data_true
    config = [0]
    s = [1.0]
    step_s = 1.0
    fu_start = 250

    indices = CartesianIndices((length(lr_start), length(l2reg), length(decay), length(decay_steps), length(steps), length(cycle), length(pat), length(config), length(s)))

    trial = Dict{Symbol,Any}(
        :lr_start=> lr_start[indices[arr].I[1]],
        :l2reg=> l2reg[indices[arr].I[2]],
        :decay => decay[indices[arr].I[3]],
        :decay_steps => decay_steps[indices[arr].I[4]],
        :steps => steps[indices[arr].I[5]],
        :cycle => cycle[indices[arr].I[6]],
        :pat=> pat[indices[arr].I[7]], 
        :U_config => config[indices[arr].I[8]],
        :s => s[indices[arr].I[9]],
        :arr => arr,
        :step_s => step_s,
        :fu_start => fu_start)

    objective(trial)

    elapsed_time = time() - t1
    println("Elapsed time: ", elapsed_time, " seconds")
    println("finished")
end

main()