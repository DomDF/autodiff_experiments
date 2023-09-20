include("neural_network_base.jl")
import MLDatasets: MNIST

# Load data
mnist_data = MNIST(split = :train); mnist_test = MNIST(split = :test)

one_dim_length = size(mnist_data.features) |> x -> x[1] * x[2]

# Reshape data
training_digits = mnist_data.features[:, :, :] |> x -> reshape(x, one_dim_length, :) |> 
    x -> Float64.(x) |> x -> Matrix(transpose(x)) |> x -> x[1:1000, :]
test_digits = mnist_test.features[:, :, :] |> x -> reshape(x, one_dim_length, :) |> 
    x -> Float64.(x) |> x -> Matrix(transpose(x))

# One-hot encode labels
digits = 0:9

training_labels = mnist_data.targets |> x -> one_hot_encode(x) |> x -> x[1:1000, :]
test_labels =  mnist_test.targets |> x -> one_hot_encode(x)

# Initialize network and select suitable activation and loss functions
digit_classifer, classifier_funs = initialise_network(one_dim_length, [32, 32], length(digits), 
                                                      output_activation = "softmax", loss = "cross_entropy") 

trained_net, training_df = train(digit_classifer, classifier_funs,
                                training_digits, training_labels,
                                a_test = test_digits, y_test = test_labels,
                                n_epochs = 10^4, scheduler = true)