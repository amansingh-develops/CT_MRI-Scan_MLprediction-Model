from torch.utils.data import DataLoader
from src.data.dataset_loader import LITSDataset


csv_path = "data/lits_train.csv"
dataset_root = "data"

dataset = LITSDataset(csv_path, dataset_root)

print("Total samples:", len(dataset))


loader = DataLoader(dataset, batch_size=4, shuffle=True)


for images, liver_masks, tumor_masks in loader:

    print("Image shape:", images.shape)
    print("Liver mask shape:", liver_masks.shape)
    print("Tumor mask shape:", tumor_masks.shape)

    break