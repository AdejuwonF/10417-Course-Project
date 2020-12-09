import torchvision.transforms as transforms
import torchvision.datasets as dset

dataset = dset.MNIST(root="", train=True, download=True,
                    transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))