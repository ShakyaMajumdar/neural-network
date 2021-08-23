module neural_network

neuron_counts = [784, 30, 16, 10]

layers = [Vector{Float64}(undef, neuron_count) for neuron_count ∈ neuron_counts]

weights = [0., [
    (rand(Float64, (next_layer_neuron_count, prev_layer_neuron_count)) .* 10) .- 5
    for (next_layer_neuron_count, prev_layer_neuron_count) ∈ zip(neuron_counts[2:end], neuron_counts)
]...]
biases = [0., [rand(neuron_count) .* 20 .- 10 for neuron_count ∈ neuron_counts[2:end]]...]


end
