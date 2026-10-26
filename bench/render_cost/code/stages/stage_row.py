#!/usr/bin/env python
"""One pose of the comparison scene through the five cumulative stage variants, as two slide sheets.

The variants and the host chain are stages/run_stages.py's (`cpu_chain`, copied without the timing);
the scene and the trajectory are ref/ref_scene.py's. Unlike run_stages.py this is not a timing run:
every pose is rendered synchronously, so there is no freshness question, and the pose of interest is
approached along the trajectory (`--lead-in` frames) so that the membrane lag of 1+2+3+4+5+6 has the
history it would have in the moving sequence. Each variant keeps its own PipelineState.

    12       stages 1+2 on the device, fixed-range apparent-T display
    123      + stage 3 (apply_optics), same fixed-range display after invert_optics
    1236     + stage 6 display branch on DN16 (AGC, gamma, DDE)
    12356    + noise (sigma_TVH and the correlated NVESD components)
    123456   + membrane lag

Everything is white-hot grey: `--palette gray` for the fixed-range display, and the Boson's own
ISP (polarity white_hot, palette gray) for the display branch.

Paths are relative to bench/render_cost/, not /tmp. Run from the repo root:

    python.sh bench/render_cost/code/stages/stage_row.py              # render + sheets
    python.sh bench/render_cost/code/stages/stage_row.py --sheet-only # re-lay out from the saved .npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[2]  # bench/render_cost
VARIANTS = ("12", "123", "1236", "12356", "123456")
TITLES = {"12": "1+2", "123": "1+2+3", "1236": "1+2+3+6", "12356": "1+2+3+5+6", "123456": "1+2+3+4+5+6"}
LABELS = {
    "12": ["Radiancia de escena + atmósfera", "rango fijo {lo:g}–{hi:g} °C"],
    "123": ["+ óptica: PSF, apertura, cos⁴, carcasa", "mismo rango fijo"],
    "1236": ["+ display: AGC, gamma y DDE", "sobre DN16 (transferencia + ADC)"],
    "12356": ["+ ruido: σ TVH por píxel", "y componentes 3D correlados"],
    "123456": ["+ retardo térmico", "de la membrana del bolómetro"],
}

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--pose", type=int, default=48, help="frame of the reference trajectory (0..119 outbound)")
p.add_argument("--lead-in", type=int, default=30, help="poses rendered before it, along the trajectory")
p.add_argument("--palette", default="gray", help="fixed-range display; gray is white-hot")
p.add_argument("--t-range-c", type=float, nargs=2, default=(15.0, 160.0))
p.add_argument("--crop-on", default="HotPlate", help="object whose upper-right corner the crop is centred on")
p.add_argument("--zoom", type=int, default=8)
p.add_argument("--flat-field", action="store_true",
               help="stage 6 with the flat-field correction on, as the comparison runs configure the camera "
                    "(default: off, as run_stages.py had it); outputs get the suffix _ff")
p.add_argument("--sheet-only", action="store_true")
args, kit_args = p.parse_known_args()

RES_DIR = HERE / "results" / "stages_ref"
SHEET_DIR = HERE / "sheets" / "stages_ref"
RES_DIR.mkdir(parents=True, exist_ok=True)
SHEET_DIR.mkdir(parents=True, exist_ok=True)
SUFFIX = "_ff" if args.flat_field else ""
NPZ = RES_DIR / f"pose{args.pose:03d}{SUFFIX}.npz"

sys.path.insert(0, str(HERE / "code" / "ref"))
import ref_scene as rs  # noqa: E402


def render() -> None:
    sys.path.insert(0, str(HERE / "code" / "demo"))
    import nave_stage as ns

    # nave_stage and ref_scene were written against /tmp/irsim_render_cost; point them here instead.
    ns.BENCH = HERE / "code"  # boot() puts BENCH / "lite" on sys.path
    ns.DEMO = RES_DIR  # the generated ss1 sensor YAML goes to RES_DIR / "out"
    rs.SCENE_YAML = RES_DIR / "_scene_ref.yaml"
    rs.write_scene(weather_csv=str(HERE / "configs" / "demo" / "nave_weather_48h.csv"),
                   environment=str(HERE / "configs" / "root" / "env_isothermal.yaml"))
    ns.OBJECTS, ns.SCENE_YAML, ns.trajectory, ns.target_name = rs.OBJECTS, rs.SCENE_YAML, rs.trajectory, rs.target_name
    ctx = ns.boot(gui=False, viewport=False, kit_args=kit_args, materials=rs.materials())

    import omni.replicator.core as rep
    import warp as wp

    import lite
    from irsim.detector.quantise import quantise
    from irsim.isp.display import run_display_branch
    from irsim.isp.palette import palette_table, quantise_display
    from irsim.optics.stage import apply_optics, invert_optics
    from irsim.pipeline.core import PipelineConfig, PipelineState
    from irsim.pipeline.detector import bolometer_lag, lag_interval_s
    from irsim_isaac.pipeline import warp_stages
    from irsim_isaac.pipeline.material_ids import labels_from_payload

    settings, lut, q = ctx["settings"], ctx["lut"], ctx["quantity"]
    camera = ctx["camera"].open(settle_frames=16)
    cfg = camera.config
    # The host stages' configuration: the camera's own, or the same with the flat field enabled.
    cfg_h = cfg if not args.flat_field else PipelineConfig.from_sensor(
        ctx["sensor"], ctx["table"], lut, sky=ctx["scene"].sky_models[ctx["band"]], atmosphere=ctx["scene"].layered,
        flat_field_enabled=True)
    spec = cfg.sensor.sensor
    for _ in range(3):
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        camera._reader.step(frames=1, rt_subframes=1)
    rp = camera._reader.render_product_path
    W, H = camera.optics.resolution
    K, BITS, DEV = cfg.supersample, spec.fpa.bit_depth, "cuda:0"
    camera._reader.detach()
    t_min_k, t_max_k = (273.15 + t for t in args.t_range_c)

    def attach(name, device=None, init=None):
        a = rep.AnnotatorRegistry.get_annotator(name, init_params=init, **({"device": device} if device else {}))
        a.attach(rp)
        return a

    annos = {"instance": attach("instance_id_segmentation", "cuda", {"colorize": False}),
             "distance": attach("DistanceToCameraSD", "cuda"), "cam": attach("camera_params")}
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=1)

    def data_of(v):
        return v.get("data", v) if isinstance(v, dict) else v

    def wrap_f32(h):
        if h.ndim == 2 and h.dtype is wp.float32:
            return h
        return wp.array(ptr=h.ptr, dtype=wp.float32, shape=tuple(h.shape[:2]), device=h.device)

    def fixed_display(radiance: np.ndarray) -> np.ndarray:
        """Stage-2-style display: apparent temperature through a fixed range, `--palette`."""
        t_app = np.asarray(lut.apparent_temperature(np.asarray(radiance, np.float32), q), np.float64)
        return palette_table(args.palette)[quantise_display((t_app - t_min_k) / (t_max_k - t_min_k))]

    gpu = lite.LiteGpu(camera, (H, W), DEV)
    out2 = wp.zeros((H, W), dtype=wp.float32, device=DEV)
    t0, period, clock, ops = camera.scene.t0_s, camera.frame_period_s, [0.0], ctx["cam_ops"]
    POS, LOOK = rs.trajectory()

    def render_sync(pos, look):
        ops.set_pose(pos, look)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1)
        payload = {c: a.get_data() for c, a in annos.items()}
        c2w = np.linalg.inv(np.asarray(payload["cam"]["cameraViewTransform"], dtype=np.float64).reshape(4, 4))
        if np.abs(c2w[3, :3] - pos).max() > 1e-4 or -c2w[2, :3] @ look < 1 - 1e-9:
            raise RuntimeError("synchronous render delivered another pose")
        return payload

    def stage12(payload) -> tuple[np.ndarray, float]:
        """Stages 1+2 on the device -> scene radiance at the aperture (host float32), and its time."""
        camera.bridge.advance_to(clock[0])
        t_s = t0 + clock[0]
        gpu.refresh(labels_from_payload(payload["instance"]), t_s)
        camera.state.t_s = t_s
        clock[0] += period
        terms = warp_stages.atmosphere_terms(cfg, camera.state)
        gpu.run(lite.wrap_ids(data_of(payload["instance"])))
        warp_stages.launch_atmosphere(gpu.out, wrap_f32(data_of(payload["distance"])), gpu.sky, terms, out2, DEV)
        wp.synchronize()
        rad = out2.numpy().copy()
        assert rad.dtype == np.float32
        return rad, t_s

    def host_chain(variant: str, rad: np.ndarray, st) -> np.ndarray:
        """Stages 3-6 on the host, run_stages.py's `cpu_chain` without the timing or the M9 chain."""
        lb_h = float(lut.lookup(st.housing_temp_k, q)[()])
        flux = apply_optics(rad, spec, lb_h, supersample=K, psf=cfg_h.psf)
        if variant == "123":
            st.advance()
            return fixed_display(invert_optics(flux, spec, lb_h))
        det, fi = cfg_h.detector, st.frame_index
        signal = det.noiseless_signal_dn(flux)
        if variant == "123456":
            signal = bolometer_lag(signal, cfg_h, st, lag_interval_s(cfg_h, st))
        if variant != "1236":
            frame = det.frame_from_signal(signal, fi, cfg_h.sensor_seed)
            signal = cfg_h.noise.apply(frame.signal_dn, frame.sigma_dn, fi, fixed_override=None)
        dn16 = quantise(signal, BITS)
        if cfg_h.flat_field is not None:
            dn16 = quantise(cfg_h.flat_field.apply(dn16), BITS)
        st.advance()
        return run_display_branch(dn16, spec.isp, BITS).display8

    states = {v: PipelineState(t_s=t0) for v in VARIANTS[1:]}
    images: dict[str, np.ndarray] = {}
    for j in range(args.pose - args.lead_in, args.pose + 1):
        rad, t_s = stage12(render_sync(POS[j], LOOK[j]))
        for v in VARIANTS[1:]:
            states[v].t_s = t_s
            images[v] = host_chain(v, rad, states[v])
    images["12"] = fixed_display(rad)

    def grey(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 3:
            if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
                raise RuntimeError("the display is not grey: check --palette and the sensor's ISP palette")
            a = a[..., 0]
        return np.ascontiguousarray(a, dtype=np.uint8)

    np.savez_compressed(NPZ, scene_radiance=rad, **{f"d{v}": grey(images[v]) for v in VARIANTS})
    meta = {"pose": args.pose, "lead_in": args.lead_in, "camera_position_m": POS[args.pose].tolist(),
            "supersample": K, "flat_field": bool(args.flat_field), "resolution": [W, H], "t_range_c": list(args.t_range_c), "palette": args.palette,
            "isp": {"polarity": spec.isp.polarity, "palette": spec.isp.palette}, "frame_period_s": period,
            "scene_radiance_min_max": [float(rad.min()), float(rad.max())]}
    (RES_DIR / f"pose{args.pose:03d}{SUFFIX}.json").write_text(json.dumps(meta, indent=1, default=str))
    print("wrote", NPZ, meta, flush=True)
    ctx["app"].close()


def sheets() -> None:
    from PIL import Image, ImageDraw, ImageFont

    z = np.load(NPZ)
    imgs = {v: z[f"d{v}"] for v in VARIANTS}
    H, W = imgs["12"].shape
    lo, hi = args.t_range_c

    def font(size, bold=False):
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", size)

    # Crop: centred on the upper-right corner of the object's projected bounding box.
    obj = next(o for o in rs.OBJECTS if o["name"] == args.crop_on)
    u, v, _ = rs.project(rs._corners(obj), rs.trajectory()[0][args.pose])
    cw, ch = W // args.zoom, H // args.zoom
    x0 = int(np.clip(round(u.max() - cw / 2), 0, W - cw))
    y0 = int(np.clip(round(v.min() - ch / 2), 0, H - ch))

    def lay_out(tiles: dict[str, Image.Image], name: str, title: str, foot: str) -> None:
        pad, gap, head, lab = 40, 24, 150, 150
        sheet = Image.new("RGB", (2 * pad + 5 * W + 4 * gap, head + H + lab + 70), (16, 16, 18))
        d = ImageDraw.Draw(sheet)
        d.text((pad, 34), title, font=font(42, True), fill=(240, 240, 240))
        labels = dict(LABELS)
        if args.flat_field:
            labels["1236"] = ["+ display: flat field, AGC, gamma, DDE", "sobre DN16 (transferencia + ADC)"]
        for i, var in enumerate(VARIANTS):
            x = pad + i * (W + gap)
            sheet.paste(tiles[var], (x, head))
            d.text((x, head + H + 14), TITLES[var], font=font(40, True), fill=(255, 255, 255))
            for k, line in enumerate(labels[var]):
                d.text((x, head + H + 70 + 36 * k), line.format(lo=lo, hi=hi), font=font(27), fill=(200, 200, 205))
        d.text((pad, sheet.height - 48), foot, font=font(22), fill=(140, 140, 146))
        sheet.save(SHEET_DIR / f"{name}.png")
        w = 1920
        sheet.resize((w, round(sheet.height * w / sheet.width)), Image.LANCZOS).save(SHEET_DIR / f"{name}_1920.png")
        print("wrote", SHEET_DIR / f"{name}.png", sheet.size, flush=True)

    foot = (f"Escena de comparación (reference_wall), pose {args.pose} del tramo de ida, cámara en movimiento. "
            f"Boson 640×512 LWIR, supersample 1, white-hot. 1+2 y 1+2+3: rango fijo {lo:g}–{hi:g} °C; desde 1+2+3+6: AGC de la cámara"
            + (", con flat field." if args.flat_field else ", sin flat field."))
    full = {}
    for var in VARIANTS:
        im = Image.fromarray(imgs[var]).convert("RGB")
        ImageDraw.Draw(im).rectangle([x0 - 1, y0 - 1, x0 + cw, y0 + ch], outline=(255, 60, 60), width=1)
        full[var] = im
    lay_out(full, f"stages_row_pose{args.pose:03d}{SUFFIX}", "Etapas acumulativas del simulador: qué añade cada una", foot)
    crops = {var: Image.fromarray(imgs[var][y0:y0 + ch, x0:x0 + cw]).resize((W, H), Image.NEAREST).convert("RGB")
             for var in VARIANTS}
    lay_out(crops, f"stages_row_pose{args.pose:03d}{SUFFIX}_crop",
            f"Recorte ×{args.zoom} de la esquina de {args.crop_on} ({obj['t_c']:g} °C sobre pared a 25 °C): el borde en cada etapa",
            f"Pose {args.pose}, Boson 640×512 LWIR, white-hot. Recorte {cw}×{ch} px en ({x0}, {y0}), ampliado sin interpolar "
            f"(recuadro rojo en la hoja completa). 1+2 y 1+2+3: rango fijo {lo:g}–{hi:g} °C; desde 1+2+3+6: AGC, "
            + ("con" if args.flat_field else "sin") + " flat field.")


if __name__ == "__main__":
    if not args.sheet_only:
        render()
    sheets()
