---
title: SMIRK-UNCC
emoji: 😊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SMIRK – 3D Facial Expressions through Analysis-by-Neural-Synthesis

Official SMIRK code running unchanged from [georgeretsi/smirk](https://github.com/georgeretsi/smirk) (CVPR 2024).

## ⚙️ First-time setup — FLAME model required

SMIRK requires the **FLAME 2020** morphable face model.  
Registration is free at <https://flame.is.tue.mpg.de/>.

After registering, add two **Space secrets** (Settings → Variables and secrets):

| Secret name      | Value                        |
|------------------|------------------------------|
| `FLAME_USERNAME` | your FLAME registration email |
| `FLAME_PASSWORD` | your FLAME password           |

Then **restart** the Space — it will download FLAME automatically on startup.

## Credits

> Retsinas et al., *3D Facial Expressions through Analysis-by-Neural-Synthesis*, CVPR 2024.
