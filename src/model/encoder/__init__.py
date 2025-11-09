from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .encoder_pi3 import EncoderPi3, EncoderPi3Cfg
from .encoder_pi3_grd import EncoderPi3Grd, EncoderPi3GrdCfg
from .encoder_pi3_pred import EncoderPi3Pred, EncoderPi3PredCfg
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
    "pi3": (EncoderPi3, None),
    "pi3_grd": (EncoderPi3Grd, None),
    "pi3_pred": (EncoderPi3Pred, None),
}

EncoderCfg = EncoderNoPoSplatCfg | EncoderPi3Cfg | EncoderPi3GrdCfg | EncoderPi3PredCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
