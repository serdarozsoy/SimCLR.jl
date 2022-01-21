# Knet installation cell: run after reloading/renaming notebook
# ENV["JULIA_CUDA_USE_BINARYBUILDER"]="false"  # Use this for faster installation, otherwise CUDA libraries will be downloaded

# using Pkg; Pkg.add("Knet")
# using Pkg; Pkg.add(["MLDatasets","PyCall","LinearAlgebra"])
# import Pkg; Pkg.add("CUDA")

using CUDA
using Knet
using MLDatasets
using PyCall
include("resnet_final2.jl")
using LinearAlgebra
using JLD2


function load_cifar_dataset()
    xtrn, ytrn = CIFAR10.traindata(UInt8);
    # WxHxCxB -> HxWxCxB 
    xtrn = permutedims(xtrn, (2, 1, 3, 4))
    xtrn, ytrn
end


py"""
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

def init():
    return transforms.Compose(
                    [   transforms.ToPILImage(),
                        transforms.RandomResizedCrop(
                            (32, 32),
                            scale=(0.08, 1.0) 
                        ),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    ]
                )


def transform_x(transform_model, x):
    return transform_model(x)
"""


function transform(x)
    return py"transform_x"(py"init"(), x)[:numpy]()
end


# To test for one sample
X_train, y_train = load_cifar_dataset();


model_main = Sequential(ResNet(depth=18, in_ch=3, out_ch=512, cifar_stem=true), Linear(512,512,f=relu), Linear(512,128))


for p in params(model_main)
    p.opt = Adam(lr=0.001)
end


epochs = 1000
train_loss = zeros(epochs);


function loss_contrastive(model, x, bs)

    LARGE_NUM = 1e9
    temperature = 0.5
    
    z = transpose(model(x))
    
    _atype = KnetArray{Float32}    
    norm_z = sqrt.(sum(abs2,z,dims=2))

    zx = z ./ norm_z

    z1 = zx[1:bs,:]
    z2 = zx[bs+1:bs*2,:]
    
    
    labels = convert(_atype, Array{Float64}(I, bs, bs*2))
    mask = convert(_atype, Array{Float64}(I, bs, bs)*LARGE_NUM)
    
    logits_aa = z1*transpose(z1)/temperature - mask 
    logits_bb = z2*transpose(z2)/temperature - mask 
    logits_ab = z1*transpose(z2)/temperature
    logits_ba = z2*transpose(z1)/temperature
    
    loss_a = sum(-labels.*logsoftmax([logits_ab logits_aa], dims=2))/bs
    loss_b = sum(-labels.*logsoftmax([logits_ba logits_bb], dims=2))/bs
    
    loss = loss_a + loss_b

    #loss = sum(z)

    
    return loss
end



_atype = KnetArray{Float32}
for epoch in 1:epochs
    # println("Epoch ", epoch)
    train_loss[epoch] = 0.0
    batch_count = 0
    dtrn = minibatch(X_train, y_train, 512, shuffle=true)
    for (batch_x, batch_y) in dtrn
        bs = size(batch_x)[4]
        batch_x1 = cat([transform(batch_x[:,:,:,i]) for i in 1:bs]..., dims=4)
        batch_x2 = cat([transform(batch_x[:,:,:,i]) for i in 1:bs]..., dims=4) 
        #x2 = convert(_atype, permutedims(batch_x1, (2, 3, 1, 4)))
        x2 = convert(_atype, permutedims(cat(batch_x1 , batch_x2, dims=4), (2, 3, 1, 4)))
        loss_main= @diff loss_contrastive(model_main, x2, bs)
        #println(loss_main)
        #print(sum(labels[:,1:bs][argmax(logits_con, dims=1)]))
        train_loss[epoch] += value(loss_main)
        batch_count += 1
        for p in params(model_main)
            g = grad(loss_main, p)
            update!(value(p), g, p.opt)
        end
    end
    train_loss[epoch] /= batch_count
    println("Train epoch: ", epoch ," ,loss: ", train_loss[epoch])
    flush(stdout)
    if mod(epoch, 100) == 0
      model_name = string("results/model_main_", epoch, ".jld2") 
      save(model_name, "model", model_main)
    end
end
