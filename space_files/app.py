"""
Gradio web UI wrapper for SMIRK (https://github.com/georgeretsi/smirk).
Original SMIRK code is completely unchanged – this file only adds a web interface.
"""

import os
import sys

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

# Ensure we're running from /app so all relative paths in SMIRK code work
os.chdir("/app")

# ── sanity check ──────────────────────────────────────────────────────────────
FLAME_PKL = "assets/FLAME2020/generic_model.pkl"
CHECKPOINT = "pretrained_models/SMIRK_em1.pt"

_missing = []
if not os.path.exists(FLAME_PKL):
    _missing.append("FLAME model (assets/FLAME2020/generic_model.pkl) — "
                     "ensure HF_TOKEN secret is set and restart the Space.")
if not os.path.exists(CHECKPOINT):
    _missing.append("SMIRK checkpoint (pretrained_models/SMIRK_em1.pt)")

# ── import SMIRK modules (original, unmodified) ───────────────────────────────
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
from demo import crop_face          # reuse the helper from unmodified demo.py

# ── device & models ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

smirk_encoder = flame = renderer = None

def load_models():
    global smirk_encoder, flame, renderer
    if _missing:
        return
    smirk_encoder = SmirkEncoder().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    enc_state = {k.replace("smirk_encoder.", ""): v
                 for k, v in ckpt.items() if "smirk_encoder" in k}
    smirk_encoder.load_state_dict(enc_state)
    smirk_encoder.eval()

    flame    = FLAME().to(DEVICE)
    renderer = Renderer().to(DEVICE)
    print(f"Models loaded on {DEVICE}.")

load_models()

# ── inference ─────────────────────────────────────────────────────────────────
def run_smirk(input_pil, crop: bool, use_generator: bool, render_orig: bool):
    if _missing:
        msg = "Cannot run – missing files:\n" + "\n".join(f"• {m}" for m in _missing)
        return None, msg

    if input_pil is None:
        return None, "Please upload an image."

    image = cv2.cvtColor(np.array(input_pil), cv2.COLOR_RGB2BGR)
    orig_h, orig_w, _ = image.shape

    kpt_mediapipe = run_mediapipe(image)

    if crop:
        if kpt_mediapipe is None:
            return None, "No face detected – try without 'Crop Face'."
        kpt_mediapipe = kpt_mediapipe[..., :2]
        tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=IMAGE_SIZE)
        cropped = warp(image, tform.inverse,
                       output_shape=(IMAGE_SIZE, IMAGE_SIZE),
                       preserve_range=True).astype(np.uint8)
        cropped_kpt = np.dot(
            tform.params,
            np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T
        ).T[:, :2]
    else:
        cropped        = image
        cropped_kpt    = kpt_mediapipe

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    t   = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(DEVICE)

    with torch.no_grad():
        outputs        = smirk_encoder(t)
        flame_out      = flame.forward(outputs)
        render_out     = renderer.forward(
            flame_out["vertices"], outputs["cam"],
            landmarks_fan=flame_out["landmarks_fan"],
            landmarks_mp=flame_out["landmarks_mp"],
        )
        rendered = render_out["rendered_img"]

    # optionally back-project to original image size
    if render_orig and crop:
        arr = (rendered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        arr = warp(arr, tform, output_shape=(orig_h, orig_w), preserve_range=True).astype(np.uint8)
        rendered = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        orig_t   = torch.tensor(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        grid = torch.cat([orig_t, rendered], dim=3)
    elif render_orig:
        orig_t   = torch.tensor(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        rendered_up = F.interpolate(rendered.cpu(), (orig_h, orig_w), mode="bilinear")
        grid = torch.cat([orig_t, rendered_up], dim=3)
    else:
        grid = torch.cat([t.cpu(), rendered.cpu()], dim=3)

    # optional SMIRK neural generator
    if use_generator:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        from src.smirk_generator import SmirkGenerator
        gen = SmirkGenerator(in_channels=6, out_channels=3,
                             init_features=32, res_blocks=5).to(DEVICE)
        gen_state = {k.replace("smirk_generator.", ""): v
                     for k, v in ckpt.items() if "smirk_generator" in k}
        gen.load_state_dict(gen_state)
        gen.eval()

        face_probs   = masking_utils.load_probabilities_per_FLAME_triangle()
        rendered_mask = 1 - (rendered == 0).all(dim=1, keepdim=True).float()
        mask_ratio, mask_ratio_mul = 0.01, 5
        npoints, _   = masking_utils.mesh_based_mask_uniform_faces(
            render_out["transformed_vertices"],
            flame_faces=flame.faces_tensor,
            face_probabilities=face_probs,
            mask_ratio=mask_ratio * mask_ratio_mul,
        )
        pmask = torch.zeros_like(rendered_mask)
        rsing  = torch.randint(0, 2, (npoints.size(0),)).to(DEVICE) * 2 - 1
        rscale = torch.rand((npoints.size(0),)).to(DEVICE) * (mask_ratio_mul - 1) + 1
        rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()
        for bi in range(npoints.size(0)):
            pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

        hull_mask   = torch.from_numpy(
            create_mask(cropped_kpt, (IMAGE_SIZE, IMAGE_SIZE))
        ).float().unsqueeze(0).to(DEVICE)
        extra_pts   = t * pmask
        masked_img  = masking_utils.masking(t, hull_mask, extra_pts, 10,
                                            rendered_mask=rendered_mask)
        gen_input   = torch.cat([rendered, masked_img], dim=1)
        with torch.no_grad():
            rec = gen(gen_input)
        grid = torch.cat([grid, rec.cpu()], dim=3)

    out = (grid.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    return out, "OK"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="SMIRK – 3D Facial Expressions") as demo:
    gr.Markdown(
        "# SMIRK: 3D Facial Expressions through Analysis-by-Neural-Synthesis\n"
        "[Paper (CVPR 2024)](https://arxiv.org/abs/2404.04104) · "
        "[Code](https://github.com/georgeretsi/smirk) · "
        "[Project Page](https://georgeretsi.github.io/smirk/)"
    )

    if _missing:
        gr.Markdown("## ⚠️ Setup required\n" +
                    "\n".join(f"- {m}" for m in _missing))

    with gr.Row():
        with gr.Column():
            inp_img     = gr.Image(type="pil", label="Input Image")
            crop_cb     = gr.Checkbox(label="Crop Face (recommended)", value=True)
            gen_cb      = gr.Checkbox(label="Use SMIRK Neural Generator", value=False)
            orig_cb     = gr.Checkbox(label="Render at Original Resolution", value=False)
            run_btn     = gr.Button("Run SMIRK", variant="primary")
        with gr.Column():
            out_img     = gr.Image(type="numpy", label="Result (input | rendered mesh)")
            status_txt  = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=[["samples/test_image2.png", True, False, False]],
        inputs=[inp_img, crop_cb, gen_cb, orig_cb],
    )

    run_btn.click(
        fn=run_smirk,
        inputs=[inp_img, crop_cb, gen_cb, orig_cb],
        outputs=[out_img, status_txt],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
