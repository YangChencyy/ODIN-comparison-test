from calData import testData_ODIN
from calMetric import metric_ODIN
from densenet import DenseNet3
from dataset import *
from models.models import MNIST_Net, Fashion_MNIST_Net, Cifar_10_Net

import argparse

data_dic = {
    'MNIST': MNIST_dataset,
    'FashionMNIST': Fashion_MNIST_dataset, 
    'Cifar_10': Cifar_10_dataset,
    'SVHN': SVHN_dataset, 
    'Imagenet_r': TinyImagenet_r_dataset,
    'Imagenet_c': TinyImagenet_c_dataset
}


data_model = {
    'MNIST': MNIST_Net,
    'FashionMNIST': Fashion_MNIST_Net, 
    'Cifar_10': Cifar_10_Net   
}


def main():
    
    parser = argparse.ArgumentParser(description="ODIN parameters")

    # Add a positional argument for the number
    parser.add_argument("InD_Dataset", type=str, help="The name of the InD dataset.")
    parser.add_argument("train_batch_size", type=int, help="train_batch_size")
    parser.add_argument("test_batch_size", type=int, help="test_batch_size")
    parser.add_argument("gpu", type=int, help="number of gpu")

    # Parse the command-line arguments
    args = parser.parse_args()


    train_set, test_set, trloader, tsloader = data_dic[args.InD_Dataset](batch_size = args.train_batch_size, 
                                                                    test_batch_size = args.test_batch_size)
    OOD_sets, OOD_loaders = [], []
    if args.InD_Dataset == 'Cifar_10':
        OOD_Dataset = ['SVHN', 'Imagenet_r', 'Imagenet_c']

        # Get all OOD datasets     
        for dataset in OOD_Dataset:
            _, OOD_set, _, OODloader = data_dic[dataset](batch_size = args.train_batch_size, 
                                                        test_batch_size = args.test_batch_size)
            OOD_sets.append(OOD_set)
            OOD_loaders.append(OODloader)

    else:
        if args.InD_Dataset == 'MNIST':
            OOD_Dataset = ['FashionMNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']
        elif args.InD_Dataset == 'FashionMNIST':
            OOD_Dataset = ['MNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']
        # Get all OOD datasets     
        for dataset in OOD_Dataset:
            _, OOD_set, _, OODloader = data_dic[dataset](batch_size = args.train_batch_size, 
                                                        test_batch_size = args.test_batch_size, into_grey = True)
            OOD_sets.append(OOD_set)
            OOD_loaders.append(OODloader)


    print("InD_dataset: ", args.InD_Dataset)
    for i in range(len(OOD_sets)):
        print("OOD: ", OOD_Dataset[i])
        if args.InD_Dataset == "Cifar_10":
            net_name = "densenet10"
            net_ODIN = torch.load("./ODIN/models/{}.pth".format(net_name))  ## TODO
            print("successfully load model", net_name)
        
        else:
            net_ODIN = data_model[args.InD_Dataset]()

        criterion_ODIN = torch.nn.CrossEntropyLoss()

        tr_l = torch.utils.data.DataLoader(train_set,
                                        batch_size=1, shuffle=True)
        ood_l = torch.utils.data.DataLoader(OOD_sets[i],
                                        batch_size=1, shuffle=True)
        
        testData_ODIN(net_ODIN, criterion_ODIN, args.gpu, tr_l, ood_l, args.InD_Dataset,
                    noiseMagnitude1 = 0.0014, temper = 1000)
        metric_ODIN(args.InD_Dataset, OOD_sets[i])

if __name__ == "__main__":
    main()