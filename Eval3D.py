import os, glob, torch,warnings
import numpy as np
from monai.metrics import DiceMetric
import nibabel as nib
from monai.data import Dataset, DataLoader
from monai.transforms import AsDiscrete, Compose,LoadImaged,  AddChanneld, ToTensord
from monai.networks.nets import UNet
warnings.filterwarnings("ignore")

data_root = '/home/nv/MRLIV/'
test_images = sorted(glob.glob(os.path.join(data_root, 'IMAGE01', '*.nii.gz')))
test_labels = sorted(glob.glob(os.path.join(data_root, 'MASK', '*.nii.gz')))
data_dicts = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]
test_files = data_dicts[106:131]
dice_metric = DiceMetric(include_background=True, reduction="mean")
def transformations():
    tes_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        ToTensord(keys=['image', 'label'])])
    return tes_transforms

tes_trans = transformations()
tes_ds = Dataset(data=test_files, transform=tes_trans)
tes_loader = DataLoader(tes_ds, batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
             strides=(2, 2, 2, 2) ).to(device)

post_pred = AsDiscrete(argmax=True,  n_classes=1)
model.load_state_dict(torch.load("/home/nv/NV1/sLUMRTL644.pth"))
with torch.no_grad():
    ii=0
    for val_data in tes_loader:
        ii=ii+1
        val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)
        with torch.cuda.amp.autocast():
            y = model(val_inputs)
        y = post_pred(y)
        y=torch.squeeze(y)
        y1 = y.detach().cpu().numpy()
        affine = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
        er = nib.Nifti1Header()
        v3=nib.Nifti1Image(y1, affine, er)
        nib.save(v3, os.path.join('/home/nv/bb/','MASK%02d.nii.gz'%ii))






