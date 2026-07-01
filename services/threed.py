"""
threed.py
توليد نموذج 3D (GLB) عن طريق Shap-E.

مهم جدًا:
- Shap-E موديل تقيل ومحتاج GPU، وعلى الـ CPU (اللي هو الافتراضي في HF Spaces المجانية)
  هياخد وقت طويل جدًا أو ممكن يفشل بسبب الميموري.
- عشان كدا الميزة دي متوقفة افتراضيًا (ENABLE_3D=false في config.py).
- لو معاكي Space بمعالج GPU (T4 مثلاً) فعّليها من الـ environment variables.
- المكتبة نفسها بتتحمل lazy (جوه الفانكشن) عشان السيرفر يشتغل عادي
  حتى لو Shap-E مش متثبتة أو الميزة متوقفة.
"""

import base64
import numpy as np

from config import OUTPUTS_DIR

_xm = None
_shap_model = None
_diffusion = None
_device = None


def _ensure_loaded():
    global _xm, _shap_model, _diffusion, _device
    if _xm is not None:
        return

    import torch
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _xm = load_model("transmitter", device=_device)
    _shap_model = load_model("text300M", device=_device)
    _diffusion = diffusion_from_config(load_config("diffusion"))


def generate_3d(prompt_3d_en: str, request_id: str = "output") -> str:
    """يرجع مسار ملف الـ .glb الناتج."""
    _ensure_loaded()

    import torch
    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh

    latents = sample_latents(
        batch_size=1,
        model=_shap_model,
        diffusion=_diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(texts=[prompt_3d_en]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    output_path = str(OUTPUTS_DIR / f"{request_id}.glb")
    for latent in latents:
        t = decode_latent_mesh(_xm, latent)
        if hasattr(t, "write_glb"):
            with open(output_path, "wb") as f:
                t.write_glb(f)
        elif hasattr(t, "tri_mesh"):
            m = t.tri_mesh()
            if hasattr(m, "write_glb"):
                with open(output_path, "wb") as f:
                    m.write_glb(f)
            else:
                import trimesh

                mesh = trimesh.Trimesh(vertices=m.verts, faces=m.faces)
                if hasattr(m, "vertex_channels") and m.vertex_channels:
                    colors = np.stack(list(m.vertex_channels.values()), axis=1)
                    if colors.shape[1] >= 3:
                        colors_255 = (np.clip(colors[:, :3], 0, 1) * 255).astype(np.uint8)
                        alpha = np.full((len(colors_255), 1), 255, dtype=np.uint8)
                        mesh.visual.vertex_colors = np.hstack([colors_255, alpha])
                mesh.export(output_path)

    return output_path


def glb_to_base64(glb_path: str) -> str:
    with open(glb_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
