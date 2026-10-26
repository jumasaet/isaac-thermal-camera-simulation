#!/usr/bin/env python
"""Why is the RGB washed out? One pose, synchronous, a handful of light/tonemap variants."""
import pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import nave_stage as ns
ctx = ns.boot(gui=False, viewport=False, kit_args=[], materials=ns.load_materials())
import numpy as np
import omni.replicator.core as rep
from pxr import UsdLux, Gf, UsdGeom
from PIL import Image
s = ctx["settings"]; stage = ctx["stage"]
camera = ctx["camera"].open(settle_frames=16)
RP = camera._reader.render_product_path; camera._reader.detach()
W, H = camera.optics.resolution
s.set("/rtx/rendermode", "RealTimePathTracing")
rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(RP)
pos, look = ns.trajectory("circular"); ctx["cam_ops"].set_pose(pos[20], look[20])
keys = ["/rtx/post/tonemap/op", "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/cm2Factor", "/rtx/post/tonemap/fNumber",
        "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/whitepoint", "/rtx/post/histogram/enabled",
        "/rtx/post/tonemap/exposureKey", "/rtx/post/tonemap/enableSrgbToGamma", "/rtx/sceneDb/ambientLightIntensity",
        "/rtx/post/aa/op", "/rtx/directLighting/sampledLighting/enabled", "/rtx/indirectDiffuse/enabled",
        "/rtx/ambientOcclusion/enabled", "/rtx/reflections/enabled", "/rtx/shadows/enabled"]
print("SET", json.dumps({k: s.get(k) for k in keys}), flush=True)
dome = UsdLux.DomeLight(stage.GetPrimAtPath("/Lights/Dome")); sun = UsdLux.DistantLight(stage.GetPrimAtPath("/Lights/Distant"))
out = ns.DEMO / "out" / "probe_rgb"; out.mkdir(parents=True, exist_ok=True)
def shot(tag, n=12):
    for _ in range(n): rep.orchestrator.step(rt_subframes=1)
    a = np.asarray(rgb.get_data()).reshape(H, W, -1)[..., :3]
    Image.fromarray(a).save(out / f"{tag}.png")
    print(f"SHOT {tag:28s} mean {a.mean():6.1f} p50 {np.median(a):5.0f} p99 {np.percentile(a,99):5.0f} sat>=250 {100*(a>=250).mean():5.1f}%", flush=True)
shot("a_as_brief")
sun.GetIntensityAttr().Set(0.0); shot("b_dome600_only")
sun.GetIntensityAttr().Set(2500.0); dome.GetIntensityAttr().Set(0.0); shot("c_distant2500_only")
dome.GetIntensityAttr().Set(600.0)
for iso in (50.0, 25.0, 12.0):
    s.set("/rtx/post/tonemap/filmIso", iso); shot(f"d_brief_filmIso{int(iso)}")
s.set("/rtx/post/tonemap/filmIso", 100.0)
for sc in (0.5, 0.25, 0.1):
    dome.GetIntensityAttr().Set(600.0*sc); sun.GetIntensityAttr().Set(2500.0*sc); shot(f"e_lights_x{sc}")
dome.GetIntensityAttr().Set(600.0); sun.GetIntensityAttr().Set(2500.0)
s.set("/rtx/post/aa/op", 3); s.set("/rtx/post/dlss/execMode", 1); shot("f_brief_aa_dlss", 24)
print("SET2", json.dumps({k: s.get(k) for k in keys}), flush=True)
ctx["app"].close()
