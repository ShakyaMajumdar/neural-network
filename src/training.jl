push!(LOAD_PATH, ".")
using NPZ
using neural_network

@doc """
Train the neural network on the entire training set.
"""
function run_training_set()
    @info "start reading training data"
    trainingimages = npzread("dataset/training_images.npy")
    traininglabels = parse.(Int, readlines(open("dataset/training_labels.txt")))
    @info "stop reading"

    network = NeuralNetwork([784, 30, 16, 10])

    for i ∈ 1:60000
        train!(network, trainingimages[i, :], traininglabels[i])
        if i % 600 == 0
            @info "$(i / 600)%"
        end
    end

    save_params(network)

end

run_training_set()
