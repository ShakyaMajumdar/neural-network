push!(LOAD_PATH, ".")
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

    network = NeuralNetwork([784, 30, 16, 10], "src/params")

    total_correct = 0
    for i ∈ 1:10000

        feed_forward!(network, testimages[i, :])
        prediction = findmax(network.layers[end])
        printstyled(
            "predict: $(prediction[2] - 1), " *
            "actual: $(testlabels[i]), " *
            "confidence(actual): $(network.layers[end][testlabels[i] + 1]), " *
            "confidence(predict): $(prediction[1]), " *
            "\n",
            color=(testlabels[i]==prediction[2]-1 ? (:green) : (:red))
        )

        if prediction[2] - 1 == testlabels[i]
            total_correct += 1
        end
    end
    println("Total Correct: $(total_correct)/10000, $(total_correct / 100)%")
end

run_test_set()