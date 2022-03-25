module Policies
using ..Environments
using ..DataStructures
using Random
using ReinforcementLearningEnvironments
using Flux

struct RandomPolicy{T<:AbstractRNG}
    rng::T
end

RandomPolicy() = RandomPolicy(Random.GLOBAL_RNG);

action(policy::RandomPolicy, env) = rand(policy.rng, action_space(env));
probability(::RandomPolicy, env, action) = 1.0/length(action_space(env));
function probability(::RandomPolicy, env) 
    a_space = action_space(env)
    p = 1.0/length(a_space)
    return ones(Float64, size(a_space)) .* p
end

struct NeuralNetworkDiscretePolicy{M}
    model::M
    num_inputs::Int
    num_outputs::Int
end

function get_log_probability_gradients(model::NeuralNetworkDiscretePolicy, states, actions_one_hot, prefactor)
    return gradient(params(model.model)) do 
        sum(sum((model.model(states) |> Flux.logsoftmax) .* actions_one_hot, dims=1) .* reshape(prefactor, 1, :))
    end
end
function get_state_array(model::NeuralNetworkDiscretePolicy, transitions::AbstractArray{SARS{S,A,R}}) where {S, A, R}
    # Fill up the state array
    input_states = zeros(eltype(S), model.num_inputs, length(transitions))
    for (i, t) in enumerate(transitions)
        input_states[:, i] .= t.state
    end
    return input_states
end
actions_onehot(transitions, action_space) = Flux.onehotbatch([t.action for t in transitions], action_space)
function get_model_gradient(env, model::NeuralNetworkDiscretePolicy, transitions::AbstractArray{SARS{S,A,R}}, prefactor) where {S, A, R}
    @assert length(transitions) == length(prefactor)
    input_states = get_state_array(model, transitions)
    as = actions_onehot(transitions, action_space(env))
    get_log_probability_gradients(model, input_states, as, prefactor)
end

function calculate_returns(transitions::AbstractArray{SARS{S,A,R}}, discount=one(R)) where {S,A,R}
    current_value = zero(R)
    returns = zeros(R, length(transitions))
    for (i, t) in enumerate(Iterators.reverse(transitions))
        current_value = t.reward + discount * current_value
        returns[end-i+1] = current_value
    end
    return returns
end


export action, probability, RandomPolicy, NeuralNetworkDiscretePolicy, get_log_probability_gradients, calculate_returns, get_model_gradient
end