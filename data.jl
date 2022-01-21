
using MLDatasets



function load_cifar_dataset()
    xtrn, ytrn = CIFAR10.traindata(UInt8);
    # WxHxCxB -> HxWxCxB 
    xtrn = permutedims(xtrn, (2, 1, 3, 4))
    xtrn, ytrn
end