from torchvision import transforms
from torch.utils.data import DataLoader
from .spectrogramDataset import SpectrogramDataset
from .acousticDataset import AcousticDataset
from utils import seed_everything

seed_everything(0)

def load_data(dataset_name="mimii", args=None):
    img_size = args.img_size
    batch_size = args.batch_size

    if dataset_name == "mimii":
        ''' Dataset (spectorgrams) from Mimii baseline '''
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        path = "../mimii_baseline/wav2image/{}dB/{}/id_{}/".format(args.num_db, args.machine_type, args.machine_id)
        dataset = SpectrogramDataset(root=path, train=True, transform=img_transform, args=args)
        testset = SpectrogramDataset(root=path, train=False, transform=img_transform, args=args)

    elif dataset_name == "ITRI_Small" or dataset_name == "ITRI_Big":
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        if args.data_type == "spectrogram":      # Dataset (spectorgrams) from ITRI recording
            path = "./data/{type}/{dataset}/{date}/{num_db}db/".format(
                type=args.data_type, dataset=dataset_name, date=args.date, num_db=args.num_db)
            dataset = SpectrogramDataset(root=path, train=True, transform=img_transform, args=args)
            testset = SpectrogramDataset(root=path, train=False, transform=img_transform, args=args)
        elif args.data_type == "acoustic":       # Dataset (acoustic images) from ITRI recording 
            path = "./data/{type}/{dataset}/{date}/".format(
                type=args.data_type, dataset=dataset_name, date=args.date)
            dataset = AcousticDataset(root=path, train=True, transform=img_transform, args=args)
            testset = AcousticDataset(root=path, train=False, transform=img_transform, args=args)

    elif dataset_name == "NYCU_Small":
        ''' Dataset (acoustic images) from small Windmill '''
        img_transform = transforms.Compose([
            transforms.CenterCrop(720),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset = AcousticDataset(root="./data/acoustic/NYCU_Small/", train=True, transform=img_transform, args=args)
        testset = AcousticDataset(root="./data/acoustic/NYCU_Small/", train=False, transform=img_transform, args=args)
        # dataset = WindDataset(root="./data/ITRI_Small/", train=True, transform=img_transform, args=args)
        # testset = WindDataset(root="./data/ITRI_Small/", train=False, transform=img_transform, args=args)

    print("Normal training data: {}".format(len(dataset.data)))
    print("Testing data: {}".format(len(testset.data)))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader