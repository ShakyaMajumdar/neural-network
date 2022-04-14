module neural_network

using LinearAlgebra
using NPZ

σ(x::AbstractFloat) = 1 / (1 + exp(-x))
σ_prime(y::AbstractFloat) = 0.25 + y * (1 - y)

struct NeuralNetwork
    neuron_counts::Vector{Int}
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    layers::Vector{Vector{Float64}}
end

@doc """
Construct a neural network with the provided neuron counts, weights and biases.
"""
NeuralNetwork(neuron_counts::Vector{Int}, weights::Vector{Matrix{Float64}}, biases::Vector{Vector{Float64}}) = NeuralNetwork(
    neuron_counts,
    weights,
    biases,
    [Vector{Float64}(undef, neuron_count) for neuron_count ∈ neuron_counts]
)

@doc """
Construct a neural network with random weights and biases.
"""
NeuralNetwork(neuron_counts::Vector{Int}) = NeuralNetwork(
    neuron_counts,
    [
        Array{Float64}(undef, 0, 0),
        [
            (rand(Float64, (next_layer_neuron_count, prev_layer_neuron_count)) .* 10) .- 5
            for (next_layer_neuron_count, prev_layer_neuron_count) ∈ zip(neuron_counts[2:end], neuron_counts)
        ]...
    ],
    [
        Array{Float64}(undef, 0),
        [rand(neuron_count) .* 20 .- 10 for neuron_count ∈ neuron_counts[2:end]]...
    ]
)


@doc """
Construct a neural network from params stored in the provided directory path.
"""
NeuralNetwork(neuron_counts::Vector{Int}, directory::String) = NeuralNetwork(
    neuron_counts,
    [npzread("$(directory)/weight$(i).npy") for i in 1:length(neuron_counts)],
    [npzread("$(directory)/bias$(i).npy") for i in 1:length(neuron_counts)]
)

@doc """
Run the neural network on the provided input image.
"""
function feed_forward!(network::NeuralNetwork, image::AbstractVector{Float64})
    network.layers[1] .= image
    @inbounds for i ∈ 2:length(network.neuron_counts)
        network.layers[i] .= σ.((network.weights[i] * network.layers[i - 1]) .+ network.biases[i])
    end
end

@doc """
Given a network run on some input, the target output for that input and the learning rate, backpropagate and rectify weights and biases.
"""
function back_propagate!(network::NeuralNetwork, target::Vector{Float64}, η::Float64)
    delc_dela::Vector{Float64} = 2 .* (network.layers[end] .- target)
    @inbounds for L ∈ length(network.neuron_counts):-1:2
        delc_delz::Vector{Float64} = delc_dela .* σ_prime.(network.layers[L])

        delc_delw::Matrix{Float64} = [j * k for j ∈ delc_delz, k ∈ network.layers[L-1]]
        delc_delb::Vector{Float64} = delc_delz
        delc_dela = network.weights[L]' * delc_delz

        # update weights and biases at this layer
        network.biases[L] .+= -η * delc_delb
        network.weights[L] .+= -η * delc_delw
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
Train the neural network with a single image and its label.
"""
function train!(network::NeuralNetwork, image::AbstractVector{Float64}, label::Int, η::Float64 = 0.05)
    feed_forward!(network, image)

    prediction = findmax(network.layers[end])
    @debug "" predict = (prediction[2] - 1) actual=label confidence_actual=network.layers[end][label + 1] confidence_predict=prediction[1]

    target = [i == label ? 1. : 0. for i ∈ 0:9]
    back_propagate!(network, target, η)
end

export NeuralNetwork, feed_forward!, back_propagate!, train!, save_params
end
