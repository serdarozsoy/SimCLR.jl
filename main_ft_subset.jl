# import Pkg; Pkg.add(["Knet","MLDatasets","PyCall","LinearAlgebra","JLD2","CUDA"])

using CUDA
using Knet
using MLDatasets
using PyCall
using LinearAlgebra
using JLD2

include("resnet.jl")
include("data.jl")
include("data_transform.jl")
include("loss.jl")

# Load pretrained model
pretrained_model_name = "results/model_main_1000.jld2"
pretrained_epoch = split(split(pretrained_model_name, ".")[1], "_" )[end]
model_trained = load( pretrained_model_name , "model")


# Only take encoder part, then add Linear Classifier
model_base = deepcopy(model_trained.layers[1])
model_ftune = Sequential(model_base, Linear(512,10))


# Parameters
optim_lr = 0.001
epochs = 50
batchsize = 256


for p2 in params(model_ftune)
    p2.opt = Adam(lr=optim_lr)
end


function nll_loss(model, x, y)
    loss = nll(model(x), y)
    return loss
end



train_loss2 = zeros(epochs);
test_loss2 = zeros(epochs);
train_acc2 = zeros(epochs);
test_acc2 = zeros(epochs);
_atype = KnetArray{Float32}

"""
Subset training dataset equally 
"""
function subset_dataset(x_train, y_train)
    subset_c = 0.1 
    t_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    subset_list = []
    cls_no = length(t_classes)
    for k = 1:cls_no
        new_list = rand([i for (i, j) in enumerate(y_train) if j == k], Int(size(x_train)[end]/cls_no*subset_c))
        push!(subset_list, new_list)
    end
    subset_list_all = collect(Iterators.flatten(subset_list))
    subset_xtrn2 = reshape(collect(Iterators.flatten([x_train[:, :, :, i] for i in subset_list_all])), (32,32,3,Int(size(xtrn2)[end]*subset_c)))
    subset_ytrn2 = y_train[subset_list_all]
    return subset_xtrn2, subset_ytrn2
end

xtrn2, ytrn2, xtst2, ytst2 = load_cifar_dataset2();
# Subset train dataset 
xtrn2, ytrn2 = subset_dataset(xtrn2, ytrn2)


_atype = KnetArray{Float32}
for epoch in 1:epochs
    train_loss2[epoch] = 0.0
    test_loss2[epoch] = 0.0
    train_acc2[epoch] = 0.0
    test_acc2[epoch] = 0.0
    batch_count = 0
    batch_count2 = 0
    # Training 
    dtrn2 = minibatch(xtrn2, ytrn2, bsize, shuffle=true)
    for (batch_x, batch_y) in dtrn2
        bs = size(batch_x)[4]
        # Transform input according to defined data augmentations
        batch_x = cat([transform(batch_x[:,:,:,i], true) for i in 1:bs]..., dims=4)
        # Permute dimension according to ResNet input dimensions
        batch_x = convert(_atype, permutedims(batch_x, (2, 3, 1, 4)))
	    loss = @diff nll_loss(model_ftune, batch_x, batch_y)
        train_loss2[epoch] += value(loss)
        batch_count += 1
        # Optimize only whole model weights
        for p2 in params(model_ftune)
            g2 = grad(loss, p2)
            update!(value(p2), g2, p2.opt)
        end
    end
    # Testing
    dtst2 = minibatch(xtst2, ytst2, bsize, shuffle=false)
    for (batch_x2, batch_y2) in dtst2
        bs2 = size(batch_x2)[4]
        # Transform input according to defined data augmentations
        batch_x2 = cat([transform(batch_x2[:,:,:,i], false) for i in 1:bs2]..., dims=4)
        # Permute dimension according to ResNet input dimensions
        batch_x2 = convert(_atype, permutedims(batch_x2, (2, 3, 1, 4)))
        scores2 =  model_ftune(batch_x2)
        loss2 = nll(scores2, batch_y2)
        test_loss2[epoch] += value(loss2)
        test_acc2[epoch] += accuracy(scores2, batch_y2)
        batch_count2 += 1
    end
    train_loss2[epoch] /= batch_count
    test_acc2[epoch] /= batch_count2
    test_loss2[epoch] /= batch_count2
    println("Train epoch: ", epoch ," ,loss: ", train_loss2[epoch])
    println("Test epoch: ", epoch ," ,loss: ", test_loss2[epoch])
    println("Test epoch: ", epoch ," ,acc: ", test_acc2[epoch])
    result_name = string("linear_results/ft_sub0.1_testacc_pre_", pretrained_epoch, ".jld2")
    result_name2 = string("linear_results/ft_sub0.1_trainloss_pre_", pretrained_epoch, ".jld2")
    result_name3 = string("linear_results/ft_sub_0.1_testloss_pre_", pretrained_epoch, ".jld2")
    save(result_name, "test_acc", test_acc2)
    save(result_name2, "train_loss", train_loss2)
    save(result_name3, "test_loss", test_loss2)
    # In batch running, program waits to finish for printing output
    flush(stdout)
    # Saving model for at the end 
    if mod(epoch, epochs) == 0
      linear_model_name = string("linear_results/ft_sub0.1_model_linear_pre_", pretrained_epoch, ".jld2") 
      save(linear_model_name, "model", model_ftune)
    end
    #GC.gc()
end
