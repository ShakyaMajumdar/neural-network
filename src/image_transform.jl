module image_transform
export image_to_vector

using Images

function image_to_vector(filepath::String)
    image = load(filepath)
    inverted_bw = broadcast(x -> 1.0 - x, Gray.(image)) # black on white to white on black
    not_black = broadcast(!=, inverted_bw, Gray{N0f8}(0.0)) # pixels which are not black
    bounding_box = (
        up = findall(any(not_black, dims=2)[:, 1])[begin],
        down = findall(any(not_black, dims=2)[:, 1])[end],
        left = findall(any(not_black, dims=1)[1, :])[begin],
        right = findall(any(not_black, dims=1)[1, :])[end]
    )
    canvas = zeros((28, 28))

    # image within boundinx box is scaled to 20x20 and pasted in the center of the canvas
    canvas[5:24, 5:24] = imresize(inverted_bw[bounding_box.up:bounding_box.down, bounding_box.left:bounding_box.right], (20, 20))
    return reduce(vcat, Float64.(canvas)')
end
end