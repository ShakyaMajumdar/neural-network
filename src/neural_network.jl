module neural_network

using LinearAlgebra
using NPZ
using Statistics

σ(x::Float64) = 1 / (1 + exp(-x))
σ_prime(y::Float64) = y * (1 - y)

struct NeuralNetwork
    neuron_counts::Vector{Int}
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    layers::Vector{Matrix{Float64}}
end

@doc """
Construct a neural network from params stored in the provided directory path.
"""
NeuralNetwork(neuron_counts::Vector{Int}, directory::String) = NeuralNetwork(
    neuron_counts,
    [npzread("$(directory)/weight$(i).npy") for i in 1:length(neuron_counts)],
    [npzread("$(directory)/bias$(i).npy") for i in 1:length(neuron_counts)],
    [Matrix{Float64}(undef, (neuron_count, 1)) for neuron_count in neuron_counts]
)

struct Trainer
    network::NeuralNetwork
    dc_dz::Vector{Matrix{Float64}}
    batch_size::Int
end

Trainer(neuron_counts, batch_size::Int) = Trainer(
    NeuralNetwork(
        neuron_counts,
        [
            Array{Float64}(undef, 0, 0),
            [
                (rand(Float64, (next_layer_neuron_count, prev_layer_neuron_count)) .* 2) .- 1
                for (next_layer_neuron_count, prev_layer_neuron_count) ∈ zip(neuron_counts[2:end], neuron_counts)
            ]...
        ],
        [
            Array{Float64}(undef, 0),
            [rand(neuron_count) .* 2 .- 1 for neuron_count ∈ neuron_counts[2:end]]...
        ],
        [Matrix{Float64}(undef, (neuron_count, batch_size)) for neuron_count in neuron_counts]
    ), 
    [Matrix{Float64}(undef, (neuron_count, batch_size)) for neuron_count in neuron_counts],
    batch_size
)

@doc """
Run the neural network on the provided input image.
"""
function feed_forward!(network::NeuralNetwork, image_batch::AbstractMatrix{Float64})
    network.layers[1] .= image_batch
    @inbounds for i ∈ 2:length(network.neuron_counts)
        network.layers[i] .= σ.((network.weights[i] * network.layers[i - 1]) .+ network.biases[i])
    end
end

@doc """
Given a network run on some input, the target output for that input and the learning rate, backpropagate and rectify weights and biases.
"""
function back_propagate!(trainer::Trainer, target::Matrix{Float64}, η::Float64)
    network = trainer.network
    trainer.dc_dz[end] .= 2 .* (network.layers[end] .- target) .* σ_prime.(network.layers[end])
    @inbounds for L ∈ length(network.neuron_counts):-1:2
        trainer.dc_dz[L-1] .= network.weights[L]' * trainer.dc_dz[L] .* σ_prime.(network.layers[L-1])
        network.weights[L] .+= -η * trainer.dc_dz[L] * network.layers[L-1]' / trainer.batch_size
        network.biases[L] .+= -η * mean(trainer.dc_dz[L], dims=2)[:, 1]
    end
end

@doc """
Save the weights and biases of the network on the disk.
"""
function save_params(network::NeuralNetwork)
    mkpath("src/params")
    @info "saving weights"
    for (i, weight) in enumerate(network.weights)
        npzwrite("src/params/weight$(i).npy", weight)
    end
    @info "saved weights"

    @info "saving biases"
    for (i, bias) in enumerate(network.biases)
        npzwrite("src/params/bias$(i).npy", bias)
    end
    @info "saved biases"
end

@doc """
Train the neural network with a single image batch and its labels.
"""
function train!(trainer::Trainer, image_batch::AbstractMatrix{Float64}, labels::Vector{Int}, η::Float64 = 0.05)
    network = trainer.network
    feed_forward!(network, image_batch)
    target = [i == label ? 1. : 0. for i ∈ 0:9, label ∈ labels]
    back_propagate!(trainer, target, η)
    return mean(norm.(eachcol(target - network.layers[end])))
end

export NeuralNetwork, Trainer, feed_forward!, back_propagate!, train!, save_params
end
