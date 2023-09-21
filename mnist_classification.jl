include("neural_network_base.jl")
import MLDatasets: MNIST

# Load data
mnist_data = MNIST(split = :train); mnist_test = MNIST(split = :test)

one_dim_length = size(mnist_data.features) |> x -> x[1] * x[2]

n_training_data = 10_000
n_training_data <= length(mnist_data.features[:, :, :] |> x -> reshape(x, one_dim_length, :))

# Reshape data
training_digits = mnist_data.features[:, :, :] |> x -> reshape(x, one_dim_length, :) |> 
    x -> Float64.(x) |> x -> Matrix(transpose(x)) |> x -> x[1:n_training_data, :]
test_digits = mnist_test.features[:, :, :] |> x -> reshape(x, one_dim_length, :) |> 
    x -> Float64.(x) |> x -> Matrix(transpose(x))

# One-hot encode labels
digits = 0:9

training_labels = mnist_data.targets |> x -> one_hot_encode(x) |> x -> x[1:n_training_data, :]
test_labels =  mnist_test.targets |> x -> one_hot_encode(x)

# Initialize network and select suitable activation and loss functions
digit_classifer, classifier_funs = initialise_network(one_dim_length, [32, 16, 8], length(digits), 
                                                      output_activation = "softmax", loss = "cross_entropy") 

# Using fewer epochs here, as this training is much slower than the regression example (esp. w/o GPU)
trained_net, training_df = train(digit_classifer, classifier_funs,
                                training_digits, training_labels,
                                a_test = test_digits, y_test = test_labels,
                                n_epochs = 100, scheduler = false, ηᵢ = 0.001)

# Plot loss
using Plots

plot(training_df.epoch, log10.(training_df.loss), alpha = 1/2, legend = :right, 
     xlabel = "Epoch", ylabel = "cross-entropy loss", label = "Training Loss", 
     title = "MNIST Classification Model")
plot!(training_df.epoch, log10.(training_df.test_loss), alpha = 1/2, label = "Test Loss")

savefig("classification_loss.png")