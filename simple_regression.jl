include("neural_network_base.jl")

# Define the function that we want the network to approximate
function gen_data(a₁::Array{Float64, 2})
    return 2 .+ 3 .* a₁[:,1] .+ 5 .* a₁[:,2]
end

# Initialize network and generate inputs (a₁) and targets (y)
mlp, mlp_funs = initialise_network(2, [8, 8], 1)
a = MersenneTwister(240819) |> prng -> randn(prng, (100, mlp.input_dim)); y = gen_data(a)

a_train = a[1:80, :]; y_train = y[1:80]
a_test = a[81:end, :]; y_test = y[81:end]

# Return the trained network parameters and the loss for each epoch
trained_net, training_df = train(mlp, mlp_funs, a_train, y_train,
                                a_test = a_test, y_test = y_test,
                                n_epochs = 10^4, scheduler = true)

using Plots

plot(training_df.epoch, log10.(training_df.loss), alpha = 1/2, 
     xlabel = "Epoch", ylabel = "Log₁₀(mse)", label = "Training Loss", title = "Simple Regression Model")
plot!(training_df.epoch, log10.(training_df.test_loss), alpha = 1/2, label = "Test Loss")

savefig("regression_loss .png")
