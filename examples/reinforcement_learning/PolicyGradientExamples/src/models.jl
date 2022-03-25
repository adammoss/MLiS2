module Models
using ..Environments
using ..Policies


using Flux


function create_cartpole_model()
    n_inputs = 4
    n_outputs = 2
    mdl = Flux.Chain(
        Dense(n_inputs, 32, Flux.selu), # Scaled exponential linear units
        Dense(32, 32, Flux.selu), # See "Self-Normalizing Neural Networks" (https://arxiv.org/abs/1706.02515).
        Dense(32, n_outputs) # logits require an identity to be anything
    )
    return NeuralNetworkDiscretePolicy(mdl, n_inputs, n_outputs)
end

export create_cartpole_model
end