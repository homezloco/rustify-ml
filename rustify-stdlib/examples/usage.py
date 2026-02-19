import rustify_stdlib as rs

# Basic usage examples
print("euclidean:", rs.euclidean([0.0, 3.0, 4.0], [0.0, 0.0, 0.0]))
print("dot_product:", rs.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
print("moving_average:", rs.moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 3))
print("convolve1d:", rs.convolve1d([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0]))
print("bpe_encode:", rs.bpe_encode("ab", []))
