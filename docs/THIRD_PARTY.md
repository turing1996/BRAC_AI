# Third-party components

This repository interfaces with or incorporates code derived from third-party projects. Their original licenses and citation requirements should be reviewed before public release.

## UNI

The tile feature extractor uses the UNI pathology foundation model from MahmoodLab. UNI weights are not included here. Access and model-loading instructions should follow the official MahmoodLab UNI distribution and its applicable terms.

## ImageNet-pretrained ViT-B/16

The downstream global H&E branch contains a ViT implementation corresponding to the `PyTorch-Pretrained-ViT` model family and expects the `B_16_imagenet1k.pth` checkpoint. Review and preserve the upstream attribution/license when releasing this vendored implementation.

## CycleGAN

The stain-normalization module implements a standard CycleGAN-style generator/discriminator training framework. If the code was adapted from a specific upstream implementation, add the original repository, license and attribution here before public release.
