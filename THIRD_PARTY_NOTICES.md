# Third-party code and provenance

## Noise2Noise PyTorch U-Net

`reproduction/models.py` adapts the U-Net topology found in the archived
`Self2Self/noise2noise/noise2noise-finger/src/unet.py`.
The archive traces back to [Joey Litalien's noise2noise-pytorch](https://github.com/joeylitalien/noise2noise-pytorch).
The MIT notice found in the archive is preserved in
`LICENSES/noise2noise-pytorch-MIT.txt` (Copyright 2018 Joey Litalien).
Local changes include configurable first pooling and skip-size alignment.
This does not establish that the archived code is the paper-final training version.

## Deep Image Prior

`reproduction/legacy_dip/*.py` are byte-for-byte copies of the archived
`deep-image-prior/models/` files. Their hashes and archive-relative locations
are in `reproduction/legacy_dip/SOURCE.json`.
Upstream project: [Dmitry Ulyanov et al., Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior).
The exact license notice found in this archive is preserved in
`LICENSES/deep-image-prior-NAS.txt`, including its additional commercial-contact text.
No broader license is asserted here.

## Other dependencies and methods

PyTorch, TensorFlow/Keras, ONNX, onnx2torch, Pillow, NumPy and their dependencies
are installed separately under their own licenses.
The separately hosted classifiers and weights are recovered experimental artifacts;
this repository does not assert new distribution rights over them.

DIPDefend source has not been imported into this repository. Its located upstream
repository is [Kira0096/DIPDefend](https://github.com/Kira0096/DIPDefend).
DIPDefend, TVM and JPEG results have not been independently reproduced here.
