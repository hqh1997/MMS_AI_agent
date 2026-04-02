from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    RandFlipd,
    Invertd,
    Resized,
    EnsureTyped,
    RandRotate90d,
    Activationsd,
    ResizeWithPadOrCropd,
)

def custom_select_fn(x):
    return x > 0
def get_transform(split):
    if split == "train":
        transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),   
        CropForegroundd(keys=["image", "label"], source_key="label", select_fn=custom_select_fn, margin=10),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=3,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
            allow_smaller =True,),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(96, 96, 96), mode=('constant')),
        ])
    elif split == "valid":
        transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="label", select_fn=custom_select_fn, margin=10),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        ])
    elif split == "post":
        transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="label", select_fn=custom_select_fn, margin=10),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ])
    else:
        raise ValueError(f"split {split} is not supported")

    return transform
