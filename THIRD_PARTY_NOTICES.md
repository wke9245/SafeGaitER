# Third-party references

SafeGait-ER's risk, protection, release, stability, and auditor components are implemented in this repository.

## IHGR-Net

The hierarchical reliability refinement follows the design principles described in:

**Towards Trustworthy Gait Recognition: Visualizing and Refining Hierarchical Motion Dependencies.** *Complex & Intelligent Systems*, 2026.

https://doi.org/10.1007/s40747-026-02381-6

The method reference informs the activation-energy gates at 1, 2, and 4 spatial granularities. The implementation is independently written from those principles; complete IHGR-Net source code is not included.

## Engineering and algorithmic references

| Project | Reference use | Upstream source |
| --- | --- | --- |
| OpenGait | Plain and focal frame convolutions, silhouette organization | https://github.com/ShiqiYu/OpenGait |
| Captum | Gradient-times-activation attribution | https://github.com/pytorch/captum |
| DeepLeak | Explanation hardening background | https://github.com/um-dsp/DeepLeak |
| Privacy Meter | Privacy auditing background | https://github.com/privacytrustlab/ml_privacy_meter |

The supplied OpenGait materials specify academic use. Third-party Python dependencies retain their respective licenses.

No dataset silhouettes, pretrained recognition weights, or complete upstream repositories are distributed here.
