push!(LOAD_PATH, ".")
using ProgressBars
using NPZ
using neural_network

@doc """
Train the neural network on the entire training set.
"""
function run_training_set()
    @info "start reading training data"
    trainingimages::Matrix{Float64} = npzread("dataset/training_images.npy")
    traininglabels::Vector{Int} = parse.(Int, readlines(open("dataset/training_labels.txt")))
    @info "stop reading"

    network::NeuralNetwork = NeuralNetwork([784, 16, 16, 10])

    for (i, image, label) ∈ zip(ProgressBar(1:60000), eachcol(trainingimages), traininglabels)
        train!(network, image, label)
    end

    save_params(network)
end

@time run_training_set()
