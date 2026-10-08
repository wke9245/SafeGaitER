# Third-party references

The method-specific protection, release, and attacker modules are maintained in this repository. The following upstream projects provide algorithmic and engineering references:

| Project | Reference use | Upstream source |
| --- | --- | --- |
| OpenGait | Frame feature extractors, data organization, benchmark protocols | https://github.com/ShiqiYu/OpenGait |
| DeepLeak | Explanation clipping, noise, and masking | https://github.com/um-dsp/DeepLeak |
| Captum | Gradient-times-activation attribution | https://github.com/pytorch/captum |
| Privacy Meter | Independent privacy attack evaluation | https://github.com/privacytrustlab/ml_privacy_meter |

The OU-MVLP identity partition in `configs/splits/OUMVLP.json` originates from the OpenGait benchmark metadata. The upstream OpenGait repository specifies academic use.

Third-party Python dependencies retain their respective licenses. No dataset silhouettes, upstream pretrained recognition weights, or complete upstream repository copies are distributed here.
