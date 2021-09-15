push!(LOAD_PATH, ".")
using ProgressBars
using NPZ
using neural_network

@doc """
Test the neural network on the test set.
"""
function run_test_set()
    @info "start reading test data"
    testimages = npzread("dataset/training_images.npy")
    testlabels = parse.(Int, readlines(open("dataset/training_labels.txt")))
    @info "stop reading"

    network = NeuralNetwork([784, 200, 75, 10], "src/params")

    total_correct = 0
    iter = ProgressBar(1:10000)
    for i ∈ iter
        feed_forward!(network, testimages[i, :])
        prediction = findmax(network.layers[end])
        if prediction[2] - 1 == testlabels[i]
            total_correct += 1
        end
        set_description(iter, "Total Correct: $(total_correct) / $(i), $(round(total_correct * 100 / i, digits=2))%")
    end
end

run_test_set()