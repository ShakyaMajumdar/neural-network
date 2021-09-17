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
    canvas_dims = (28, 28)
    canvas = zeros((canvas_dims))

    bounding_box_dims = (bounding_box.down - bounding_box.up, bounding_box.right - bounding_box.left)
    rescaled_height, rescaled_width = round.(Int, (bounding_box_dims) .* (20 / maximum(bounding_box_dims)))

    # image within bounding box is scaled (maintaining aspect ratio) to fit in 20x20 and pasted in the center of the canvas
    canvas[
        canvas_dims[1]÷2 - rescaled_height÷2 : canvas_dims[1]÷2 + (rescaled_height - rescaled_height÷2) - 1,
        canvas_dims[2]÷2 - rescaled_width÷2 : canvas_dims[2]÷2 + (rescaled_width - rescaled_width÷2) - 1,
        ] = imresize(
        inverted_bw[bounding_box.up:bounding_box.down, bounding_box.left:bounding_box.right]
            |> dilate
            |> dilate
            |> dilate,
        (rescaled_height, rescaled_width)
    )

    return reduce(vcat, Float64.(canvas)')
end
end