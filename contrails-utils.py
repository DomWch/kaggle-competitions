# %% [code]
import sys

sys.path.append("../input/pretrained-models-pytorch")
sys.path.append("../input/efficientnet-pytorch")
sys.path.append("/kaggle/input/smp-github/segmentation_models.pytorch-master")
sys.path.append("/kaggle/input/timm-pretrained-resnest/resnest/")
import segmentation_models_pytorch as smp

import yaml
from pathlib import Path
from typing import Literal
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as T
import pytorch_lightning as pl
from torchmetrics.functional import dice
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import cartopy.crs as ccrs
from shapely import wkt
from IPython import display
from matplotlib import animation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


class npfast:
    ### from https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/414549
    def load(file):
        file = open(file, "rb")
        header = file.read(128)
        descr = str(header[19:25], "utf-8").replace("'", "").replace(" ", "")
        shape = tuple(
            int(num)
            for num in str(header[60:120], "utf-8")
            .replace(", }", "")
            .replace("(", "")
            .replace(")", "")
            .split(",")
        )
        datasize = np.lib.format.descr_to_dtype(descr).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))


class Load:
    def open_record(record_path: Path, CFG):
        bands = [
            npfast.load(record_path / f"band_{band_num:02d}.npy")
            for band_num in CFG.band_interval
        ]
        pixel_mask = npfast.load(record_path / CFG.model.target)
        return np.array(bands), pixel_mask

    def false_color(band11, band14, band15):
        """
        convert bands to rgb that labelers saw
        """

        def normalize(band, bounds):
            return (band - bounds[0]) / (bounds[1] - bounds[0])

        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)

        r = normalize(band15 - band14, _TDIFF_BOUNDS)
        g = normalize(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize(band14, _T11_BOUNDS)

        return np.clip(np.stack([r, g, b], axis=2), 0, 1)

    def open_record_false_color(
        record: Path,
        N_TIMES_BEFORE: int = 4,
    ):
        band11 = npfast.load(record / "band_11.npy")[..., N_TIMES_BEFORE]
        band14 = npfast.load(record / "band_14.npy")[..., N_TIMES_BEFORE]
        band15 = npfast.load(record / "band_15.npy")[..., N_TIMES_BEFORE]
        return np.array(Load.false_color(band11, band14, band15))

    def open_record_all_sequence(record_path: Path, pixel_mask=True):
        band11 = npfast.load(record_path / "band_11.npy")
        band14 = npfast.load(record_path / "band_14.npy")
        band15 = npfast.load(record_path / "band_15.npy")
        if pixel_mask == True:
            pixel_mask = npfast.load(record_path / "human_pixel_masks.npy")
        return (
            Load.false_color(band11, band14, band15),
            pixel_mask,
        )


class Commun:
    class colors:
        """Ex colors.BOLD.format(str)"""

        PURPLE = "\033[95m{}\033[0m"
        BLUE = "\033[94m{}\033[0m"
        CYAN = "\033[96m{}\033[0m"
        GREEN = "\033[92m{}\033[0m"
        YELLOW = "\033[93m{}\033[0m"
        RED = "\033[91m{}\033[0m"
        BOLD = "\033[1m{}\033[0m"
        UNDERLINE = "\033[4m{}\033[0m"

    def run_length_encode(x: np.array, fg_val_min=1) -> str:
        dots = np.where((x.T.flatten() == fg_val_min))[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if b > prev + 1:
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        #     print(run_lengths)
        return (
            " ".join([str(run_l) for run_l in run_lengths])
            if len(run_lengths) > 0
            else "-"
        )

    def run_length_decode(mask_rle: str, shape=(256, 256)) -> np.array:
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        if mask_rle != "-":
            s = mask_rle.split()
            starts, lengths = [
                np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
            ]
            starts -= 1
            ends = starts + lengths
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
        return img.reshape(shape, order="F")  # Needed to align to RLE direction


class Visualization:
    PseudoPathDefault = Path(
        "/kaggle/input/pseudolabelling/pseudolabel/train/pseudolabel"
    )
    TrainPath = Path(
        "/kaggle/input/google-research-identify-contrails-reduce-global-warming/train"
    )

    def plot_crs(df, central_meridian):
        data_crs = ccrs.TransverseMercator(
            central_longitude=central_meridian,
            central_latitude=0,
            false_easting=500000,
            false_northing=0.0,
            scale_factor=0.9996,
            globe=ccrs.Globe(datum="WGS84", ellipse="WGS84"),
            approx=False,
        )
        plt.figure(figsize=(30, 15))
        ax = plt.axes(projection=data_crs)
        ax.stock_img()
        ax.gridlines()
        df_tmp = df.loc[df.central_meridian == central_meridian]
        for i, row in df_tmp.reset_index(drop=True).iterrows():
            #     print(f"plot {row.row_min}, {row.col_min}")
            #     plt.plot([row.row_min, row.col_min, row.row_size, row.col_size], transform=data_crs, color='black', linewidth=1, marker='.')
            ax.add_patch(
                Rectangle(
                    xy=[row.col_min, row.row_min],
                    width=row.col_size,
                    height=row.row_size,
                    facecolor="none",
                    edgecolor="red" if row["from"] == "val" else "green",
                    transform=data_crs,
                )
            )
        #     plt.plot([row.col_min, row.row_min, row.col_size, row.row_size,], transform=data_crs, color="red")
        print(
            f'{i+1}records {len(df_tmp.loc[df_tmp["from"]=="val"])}validation {len(df_tmp.loc[df_tmp["from"]=="train"])}train \t\t du {df_tmp.timestamp.min()} au {df_tmp.timestamp.max()}'
        )

        plt.show()

    def plot_merge_crs(df):
        CRS_PARAMS = dict(
            central_latitude=0,
            false_easting=500000,
            false_northing=0.0,
            scale_factor=0.9996,
            globe=ccrs.Globe(datum="WGS84", ellipse="WGS84"),
            approx=False,
        )
        plt.figure(figsize=(30, 15))
        ax = plt.axes(
            projection=ccrs.TransverseMercator(
                central_longitude=df.central_meridian.iloc[0], **CRS_PARAMS
            )
        )
        ax.stock_img()
        ax.gridlines()

        for central_meridian in df.central_meridian.unique():
            print(f"plot map for central_longitude {central_meridian}")
            data_crs = ccrs.TransverseMercator(
                central_longitude=central_meridian, **CRS_PARAMS
            )
            df_tmp = df.loc[df.central_meridian == central_meridian]
            for i, row in df_tmp.reset_index(drop=True).iterrows():
                ax.add_patch(
                    Rectangle(
                        xy=[row.col_min, row.row_min],
                        width=row.col_size,
                        height=row.row_size,
                        facecolor="none",
                        edgecolor="red" if row["from"] == "val" else "green",
                        transform=data_crs,
                    )
                )

            print(
                f'{i+1}records {len(df_tmp.loc[df_tmp["from"]=="val"])}validation {len(df_tmp.loc[df_tmp["from"]=="train"])}train \t\t du {df_tmp.timestamp.min()} au {df_tmp.timestamp.max()}'
            )

        plt.show()

    def plot_label_sequence(record_id, pseudo_label_path=PseudoPathDefault):
        sequences = list((pseudo_label_path / str(record_id)).iterdir())
        sequences.sort(key=lambda path: path.name)
        len_sequences = len(sequences)
        plt.figure(figsize=(20, 30))
        for indice, path in enumerate(sequences):
            label = npfast.load(path / "label.npy")
            axs = plt.subplot(1, len_sequences, indice + 1)
            axs.title.set_text(
                f"predicted mask\n{path.name}\n({indice+1}/{len_sequences})"
            )
            axs.imshow(label, interpolation="none")
        plt.show()

    def plot_anim(record_id, path=TrainPath, pseudo_label_path=PseudoPathDefault):
        # target mask vs images
        images, human_mask = Load.open_record_all_sequence(path / f"{record_id}")
        plt.figure(figsize=(20, 30))
        ax1 = plt.subplot(1, 2, 1)
        ax1.title.set_text("pixel mask")
        ax1.imshow(human_mask, interpolation="none")
        ax2 = plt.subplot(1, 2, 2)
        ax2.title.set_text("input image")
        ax2.imshow(images[..., 4].astype("float32"))
        plt.show()

        # plot pseudo label
        Visualization.plot_label_sequence(record_id, pseudo_label_path)

        # animation images
        fig = plt.figure(figsize=(15, 15))
        im = plt.imshow(images[..., 0])

        def draw(i):
            im.set_array(images[..., i])
            return [im]

        anim = animation.FuncAnimation(
            fig, draw, frames=images.shape[-1], interval=500, blit=True
        )
        #     plt.title("Animation sequences images")
        plt.close()
        return display.HTML(anim.to_jshtml())


class PyLModel:
    class CFG:
        def default_train_config(image_size=384):
            return {
                "seed": 42,
                "train_bs": 128,
                "valid_bs": 128,
                "workers": 1,
                "title": "default",
                "early_stop": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 5,
                    "verbose": 1,
                },
                "trainer": {
                    "max_epochs": 10,
                    "min_epochs": 8,
                    "precision": "16-mixed",
                    "devices": 1,
                },
                "model": {
                    "seg_model": "Unet",
                    "encoder_name": "timm-resnest26d",
                    "loss_smooth": 1.0,
                    "image_size": image_size,
                    "optimizer_params": {"lr": 0.00005, "weight_decay": 0.0},
                    "scheduler": {
                        "name": "ReduceLROnPlateau",
                        "params": {
                            "CosineAnnealingLR": {
                                "T_max": 2,
                                "eta_min": 1e-06,
                                "last_epoch": -1,
                            },
                            "ReduceLROnPlateau": {
                                "mode": "min",
                                "factor": 0.5,
                                "patience": 1,
                                "verbose": True,
                            },
                        },
                    },
                },
            }

        def default_predict_config(image_size: Literal[256, 384, 512] = 384):
            return {
                "batch_size": 128,
                "workers": 1,
                "model": {
                    "seg_model": "Unet",
                    "encoder_name": "timm-resnest26d",
                    "loss_smooth": 1.0,
                    "image_size": image_size,
                },
            }

        def write_config(config: dict, model_path: Path):
            with open(model_path / "config.yaml", "w") as file:
                yaml.dump(config, file)

        def open_config(config_path: Path = Path("empty"), image_size=256):
            if config_path.exists():
                with open(config_path / "config.yaml", "r") as file_obj:
                    config = yaml.safe_load(file_obj)
            else:
                config = PyLModel.CFG.default_train_config(image_size)
            return config

    class ContrailsDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            df,
            mode: Literal["train", "pseudo_train", "validation", "test"],
            image_size=256,
        ):
            self.df = df
            self.mode = mode
            self.normalize_image = T.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
            self.image_size = image_size
            if image_size != 256:
                self.resize_image = T.transforms.Resize(image_size, antialias=True)

        def __getitem__(self, index):
            row = self.df.iloc[index]
            img = Load.open_record_false_color(
                row.path, N_TIMES_BEFORE=row.sequence_num
            )

            img = (
                torch.tensor(np.reshape(img, (256, 256, 3)))
                .to(torch.float32)
                .permute(2, 0, 1)
            )

            if self.image_size != 256:
                img = self.resize_image(img)

            img = self.normalize_image(img)

            if self.mode != "test":
                label = np.array(npfast.load(row.target))
            else:
                label = torch.tensor(int(row.record_id))
            return img.float(), label

        def __len__(self):
            return len(self.df)

    class LightningModule(pl.LightningModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = smp.Unet(
                encoder_name=config.get("encoder_name", "timm-resnest26d"),
                encoder_weights=config.get("encoder_weights", None),
                in_channels=3,
                classes=1,
                activation=None,
            )
            self.loss_module = smp.losses.DiceLoss(
                mode="binary", smooth=config.get("loss_smooth", 1)
            )
            self.val_step_outputs = []
            self.val_step_labels = []

        def forward(self, batch):
            return self.model(batch)

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.config.get("optimizer_params", {})
            )

            if self.config["scheduler"]["name"] == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    **self.config["scheduler"]["params"]["CosineAnnealingLR"],
                )
                lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
            elif self.config["scheduler"]["name"] == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **self.config["scheduler"]["params"]["ReduceLROnPlateau"],
                )
                lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
            elif (
                self.config["scheduler"]["name"]
                == "cosine_with_hard_restarts_schedule_with_warmup"
            ):
                scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer,
                    **self.config["scheduler"]["params"][
                        self.config["scheduler"]["name"]
                    ],
                )
                lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

        def training_step(self, batch, batch_idx):
            imgs, labels = batch
            preds = self.model(imgs)
            if self.config["image_size"] != 256:
                preds = torch.nn.functional.interpolate(
                    preds, size=256, mode="bilinear"
                )
            loss = self.loss_module(preds, labels)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=16,
            )

            for param_group in self.trainer.optimizers[0].param_groups:
                lr = param_group["lr"]
            self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)

            return loss

        def validation_step(self, batch, batch_idx):
            imgs, labels = batch
            preds = self.model(imgs)
            if self.config["image_size"] != 256:
                preds = torch.nn.functional.interpolate(
                    preds, size=256, mode="bilinear"
                )
            loss = self.loss_module(preds, labels)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.val_step_outputs.append(preds)
            self.val_step_labels.append(labels)

        def on_validation_epoch_end(self):
            all_preds = torch.cat(self.val_step_outputs)
            all_labels = torch.cat(self.val_step_labels)
            all_preds = torch.sigmoid(all_preds)
            self.val_step_outputs.clear()
            self.val_step_labels.clear()
            val_dice = dice(all_preds, all_labels.long())
            self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
            if self.trainer.global_rank == 0:
                print(f"\nEpoch: {self.current_epoch}", flush=True)

    def train_from_checkpoint(
        df_train,
        df_val,
        config={},
        model_path=Path(
            "/kaggle/input/identify-contrails-models/models_kaggle_contrails/pb_model_dice_649.ckpt"
        ),
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_path.exists():
            title = config.get("title", model_path.stem)
            model = PyLModel.LightningModule(config["model"]).load_from_checkpoint(
                model_path, config=config["model"], map_location=device
            )
        else:
            title = f'from_scratch_{config.get("encoder_name","default_encoder")}_{config.get("encoder_weights","default_weights")}'
            model = PyLModel.LightningModule(config["model"])
        print(f"{title}: {len(df_train)}train {len(df_val)}validation")

        early_stop_callback = pl.callbacks.EarlyStopping(**config["early_stop"])

        dataset_train = PyLModel.ContrailsDataset(df_train, mode="pseudo_train")
        dataset_validation = PyLModel.ContrailsDataset(df_val, mode="validation")

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config.get("train_bs", 128),
            shuffle=True,
            num_workers=config.get("workers", 1),
        )
        data_loader_validation = torch.utils.data.DataLoader(
            dataset_validation,
            batch_size=config.get("valid_bs", 64),
            shuffle=False,
            num_workers=config.get("workers", 1),
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_weights_only=config.get("save_weights_only", True),
            monitor="val_dice",
            dirpath=config.get("output_dir", "models"),
            mode="max",
            filename=f"model-{title}-{{val_dice:.4f}}",
            save_top_k=1,
            verbose=1,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=pl.loggers.CSVLogger(save_dir=f"logs_{title}/"),
            **config["trainer"],
        )

        trainer.fit(
            model, data_loader_train, data_loader_validation
        )  # , ckpt_path = model_path

        return model

    def predict_with_models(
        df,
        models: list[Path] = list(
            Path("/kaggle/input/training-with-4-folds/models").glob("*.ckpt")
        ),
        config={},
    ):
        gc.enable()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_preds = {}
        for i, model_path in enumerate(models):
            print(f"Predict with model: {model_path.name}")
            if config == {}:
                config = PyLModel.CFG.open_config(
                    config_path=model_path.parent / "config.yaml"
                )

            dataset = PyLModel.ContrailsDataset(
                df, mode="test", image_size=config["model"]["image_size"]
            )
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.get("batch_size", 128),
                num_workers=config.get("num_workers", 1),
            )

            model = PyLModel.LightningModule(config["model"]).load_from_checkpoint(
                model_path, config=config["model"], map_location=device
            )
            model.to(device)
            model.eval()

            model_preds = {}
            for images, image_id in data_loader:
                images = images.to(device)

                with torch.no_grad():
                    predicted_mask = model(images[:, :, :, :])
                if config["model"]["image_size"] != 256:
                    predicted_mask = torch.nn.functional.interpolate(
                        predicted_mask, size=256, mode="bilinear"
                    )
                predicted_mask = torch.sigmoid(predicted_mask).cpu().detach().numpy()

                for img_num in range(0, images.shape[0]):
                    current_mask = predicted_mask[img_num, :, :, :]
                    current_image_id = image_id[img_num].item()
                    model_preds[current_image_id] = current_mask
            all_preds[f"f{i}"] = model_preds

            del model
            torch.cuda.empty_cache()
            gc.collect()

        return all_preds, (i + 1)

    def stack_mean_prediction_mask_threshold(
        all_preds: dict, index: int, threshold: float, models_len: int
    ):
        predicted_mask = sum([preds[index] for preds in all_preds.values()]) / float(
            models_len
        )
        predicted_mask_with_threshold = np.zeros((256, 256))
        predicted_mask_with_threshold[predicted_mask[0, :, :] > threshold] = 1
        return predicted_mask_with_threshold

    def double_thresholds(image, high_threshold, low_threshold):
        image = torch.from_numpy(image)
        X = image > high_threshold
        Y = (image > low_threshold) & (image <= high_threshold)
        while True:
            X_neighborhood = torch.nn.functional.max_pool2d(
                X.float(), kernel_size=3, stride=1, padding=1
            )
            new_X = X | (Y & X_neighborhood.byte())
            if torch.all(X == new_X):
                break
            X = new_X
        result = X.float()
        return result  # returns a binary image

    def stack_prediction_mask_threshold(
        all_preds: dict,
        index: int,
        high_threshold: float,
        low_threshold: float,
        models_len: int,
        vote_majority: float = 0.45,
    ):
        # apply dual threshold before stacking, so models can have différent threshold and its a vote system more like human_individual_masks -> human_pixel_masks
        predicted_mask = sum(
            [
                PyLModel.double_thresholds(
                    preds[index],
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                )
                for preds in all_preds.values()
            ]
        )
        predicted_mask = predicted_mask > (models_len * vote_majority)
        return predicted_mask
