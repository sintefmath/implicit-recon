import os, glob, math, torch
from monai.data import DataLoader, Dataset
from monai.transforms import AsDiscrete, Compose, LoadImaged, AddChanneld, ToTensord
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism

data_root = '/home/nv/MRLIV/'
train_images = sorted(glob.glob(os.path.join(data_root, 'IMAGE01', '*.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(data_root, 'MASK01', '*.nii.gz')))
data_dicts = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:80], data_dicts[80:106]

def transformations():
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        ToTensord(keys=['image', 'label'])    ])
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        ToTensord(keys=['image', 'label'])    ])
    return train_transforms, val_transforms

def train_process(fast=False):
    epoch_num = 10
    val_interval = 1
    train_trans, val_trans = transformations()
    train_ds = Dataset(data=train_files, transform=train_trans)
    val_ds = Dataset(data=val_files, transform=val_trans)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n1=16
    model = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(n1 * 1, n1 * 2, n1 * 4, n1 * 8, n1 * 16),
                 strides=(2, 2, 2, 2)).to(device)
    loss_function = DiceLoss(to_onehot_y=True,softmax=True)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = list()
    metric_values = list()

    for epoch in range(epoch_num):
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0 :
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                    val_outputs = model(val_inputs)
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    epochs_no_improve = 0
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    torch.save(model.state_dict(), 'sLUMRTL644.pth')
                else:
                    epochs_no_improve += 1

            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                  f" best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    return epoch_num, epoch_loss_values, metric_values,  best_metrics_epochs_and_time

set_determinism(seed=0)

epoch_num, m_epoch_loss_values, m_metric_values,  m_best = train_process(fast=False)




