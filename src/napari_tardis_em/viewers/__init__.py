from tardis_em.utils.losses import *

colormap_for_display = "red"

face_colormap = "Spectral"

IMG_FORMAT = (".tif", ".am", ".mrc", ".rec", ".map")

loss_functions = [
    AdaptiveDiceLoss,
    BCELoss,
    WBCELoss,
    BCEDiceLoss,
    CELoss,
    DiceLoss,
    ClDiceLoss,
    ClBCELoss,
    SigmoidFocalLoss,
    LaplacianEigenmapsLoss,
    BCEMSELoss,
]
