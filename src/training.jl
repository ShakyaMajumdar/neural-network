push!(LOAD_PATH, ".")
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

    network::NeuralNetwork = NeuralNetwork([784, 200, 75, 10])

    for i ∈ 1:60000
        train!(network, trainingimages[i, :], traininglabels[i])
        if i % 600 == 0
            @info "$(i / 600)%"
        end
    end

    save_params(network)

end

@time run_training_set()
