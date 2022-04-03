using Random
using Base.Iterators: product
using ProgressBars
using Plots
using Statistics
gr()

function initialise_state(batch_size, max_sequence_length)
    zeros(Int, batch_size, max_sequence_length)
end

function initialise_params(max_sequence_length; type=Float64, initial_value=zero(type))
    params = zeros(type, max_sequence_length, max_sequence_length)
    params .= initial_value
    return params
end

function get_probability(params, t)
    preferences = view(params, 1:t, t)
    return inv.(ones(eltype(preferences), size(preferences)) .+ exp.(-preferences))
end


function convert_index(state, t)
    ((state + t) + 1) ÷ 2
end
function convert_index(state::AbstractArray, t)
    ((state .+ t) .+ 1) .÷ 2
end

function get_batch!(state, actions, rewards, params, cache; rng=Random.GLOBAL_RNG, scalar_reward=10.0)
    _, max_time = size(state)
    
    state[:, 1] .= zero(eltype(state))

    for t=2:max_time
        # List of probabilities for each possible state at time t-1
        probs = get_probability(params, t-1)
        # Map the states to the action probabilities, using the previous state and time
        prob_actions = map(x->probs[x], convert_index(state[:, t-1], t-1))
        # Sample the actions according to probs of model -> 1 is up and 0 is down
        rand!(rng, cache)
        actions[:, t-1] .= (cache .< prob_actions)

        rewards[:, t-1] .= -log.(actions[:, t-1].*prob_actions.+(one(eltype(actions)).-actions[:, t-1]).*(one(eltype(prob_actions)).-prob_actions))
        state[:, t] .= state[:, t-1] .+ (actions[:, t-1] .* convert(eltype(state), 2) .- one(eltype(state)))
    end
    rewards[:, max_time-1] .-= scalar_reward.*abs.(state[:, max_time])

    nothing
end

function initialise_variables(batch_size, T)
    state = initialise_state(batch_size, T+1)
    params = initialise_params(T)
    actions = zeros(Bool, batch_size, T)
    rewards = zeros(eltype(params), batch_size, T)
    cache = zeros(eltype(params), batch_size)
    return state, actions, rewards, params, cache
end

function get_grad_probability(params)
    e = exp.(-params)
    return e ./ (one(eltype(e)) .+ e) .^ 2
end

function get_grad_log_probability(param)
    return inv(one(typeof(param))+exp(param))
end

function calculate_gradients(states, actions, params, rewards; discount=one(eltype(rewards)))
    # Make returns from the rewards
    batch_size, sequence_length = size(rewards)
    returns = similar(rewards)
    returns[:, end] .= rewards[:, end] # Value of terminal state is zero
    for t = sequence_length-1:-1:1
        returns[:, t] .= rewards[:, t] .+ discount .* returns[:, t+1]
    end

    gradients = similar(params)
    gradients .= 0

    for i = 1:batch_size
        for t=1:sequence_length
            state_idx = convert_index(states[i, t], t)
            p = params[state_idx, t]

            gradients[state_idx, t] += (actions[i, t]*2-1) * returns[i, t] * get_grad_log_probability(p)
        end
    end
    
    return gradients ./ (batch_size)
end

function run_epoch!(state, actions, rewards, params, cache; lr=0.01, discount=1.0, scalar_reward=100.0, rng=Random.GLOBAL_RNG)
    get_batch!(state, actions, rewards, params, cache; scalar_reward, rng)
    gradients = calculate_gradients(state, actions, params, rewards; discount)
    params .+= lr .* gradients
end

function train!(state, actions, rewards, params, cache; epochs=100, show_progress=true, kwargs...)
    average_returns = zeros(epochs)
    batch_size, _ = size(rewards)
    iter = show_progress ? ProgressBar(1:epochs) : (1:epochs)
    for i in iter
        run_epoch!(state, actions, rewards, params, cache; kwargs...)
        average_returns[i] = sum(rewards)/batch_size
    end
    

    return average_returns
end

function get_return_curves(n_repeats = 32; T=10, batch_size=128, show_progress=true, kwargs...)
    
    iter = show_progress ? ProgressBar(1:n_repeats) : (1:n_repeats)
    returns = Vector{Any}(undef, length(iter))
    Threads.@threads for i in iter
        state, actions, rewards, params, cache = initialise_variables(batch_size, T)
        rs = train!(state, actions, rewards, params, cache; show_progress=false, kwargs...)
        returns[i] = rs
    end
    return returns
end

function plot_return_curves(returns; kwargs...)
    n_samples = length(returns)
    average_return = reshape(mean(hcat(returns...), dims=2), :)
    # Unbiased error on the mean
    error_on_mean = reshape(std(hcat(returns...), dims=2), :)

    plt = plot(1:length(average_return), average_return; ribbon=error_on_mean, legend=false, xscale=:log10, kwargs...)
    xlims!(1, length(average_return))
    xlabel!("Epochs")
    ylabel!("Average Return")    
    return plt
end

function conv(x, y)
    @assert length(y) % 2 == 1
    out = similar(x)
    w = length(y)
    halfw = w ÷ 2

    for i in 1:length(x)
        left_width = i-max(1, i-halfw)
        right_width = min(length(x), i+halfw) - i

        idx = (i-left_width):(i+right_width)
        out[i] = dot(view(x, idx), view(y, halfw-(left_width)+1:halfw+(right_width)+1))
    end
    return out
end