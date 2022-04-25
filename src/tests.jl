push!(LOAD_PATH, ".")
using ProgressBars
using NPZ
using neural_network

@doc """
Test the neural network on the test set.
"""
function run_test_set()
    @info "start reading test data"
    testimages::Matrix{Float64} = npzread("dataset/test_images.npy")
    testlabels::Vector{Int} = parse.(Int, readlines(open("dataset/test_labels.txt")))
    @info "stop reading"

    network::NeuralNetwork = NeuralNetwork([784, 16, 16, 10], "src/params")

    total_correct = 0
    iter = ProgressBar(1:10000)
    for (i, image, label) ∈ zip(iter, eachcol(testimages), testlabels)
        feed_forward!(network, image)
        prediction = argmax(network.layers[end]) - 1
        if prediction == label
            total_correct += 1
        end
        set_description(iter, "Total Correct: $(total_correct) / $(i), $(round(total_correct * 100 / i, digits=2))%")
    end
end

run_test_set()