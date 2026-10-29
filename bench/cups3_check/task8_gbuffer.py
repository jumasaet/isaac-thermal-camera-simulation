#!/usr/bin/env python
"""Task 8 (Kit, once): dump the three-cup G-buffer, so the table-reflection work runs without Kit.

    env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
        /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task8_gbuffer.py

The scene, pose, roll, YAML and warm-up are task2_render.py's, imported, not copied. The camera does not move
and every temperature is constant (the mean of the two probe readings), so one G-buffer stands for all eight
instants; task8_reflection.py replays the camera chain over it (refl_common.py).

Writes out/sim/gbuffer.npz at the render product's resolution (2560 x 1920, supersample 4), float32 where a
temperature or distance flows (CLAUDE.md rule 2):
  temperature_k, material_id, sky_mask -- the planes Clock.frame hands run_frame;
  distance_m -- RAW (DistanceToCameraSD, sky 0), WITHOUT the 0.3 m CUPS_MIN_DISTANCE_M floor: the floor is a
                defocus workaround and is applied again at replay; the reflection needs the true distance;
  ids        -- instance ids; names/ids give the prim name of each;
  view, proj -- camera_params' cameraViewTransform and cameraProjection (row-major 4 x 4, USD convention).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import task2_render as t2

c3, cm = t2.c3, t2.cm


def main() -> None:
    t2.patch_library()
    t2.patch_stage()
    t2.patch_roll(t2.POSE["camera_roll_deg"])
    objs, thermal = t2.objects()
    scene_yaml = t2.write_scene(thermal, t2.write_weather())
    th = np.radians(t2.POSE["camera_pitch_up_deg"])
    pos = np.array([0.0, 0.0, t2.POSE["camera_height_above_table_m"]])
    look = np.array([-np.cos(th), 0.0, np.sin(th)])
    ctx = cm.boot(
        sensor_yaml=t2.SENSOR,
        scene_yaml=scene_yaml,
        objects=objs,
        materials={n: m for n, (m, _t) in thermal.items()},
        first_pose=(pos, look),
    )
    import lite

    from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths

    clock = cm.Clock(ctx)
    for _ in range(4):
        clock.frame(cm.render_sync(ctx, pos, look))
    clock.rel = t2.S0 + float(t2.TR[t2.RENDER_IDX[0]])
    payload = cm.render_sync(ctx, pos, look)
    hs, ws = ctx["shape_ss"]
    camera = ctx["camera"]
    camera.bridge.advance_to(clock.rel)
    t_s = clock.t0 + clock.rel
    ids = cm._data(payload["instance"]).numpy().reshape(hs, ws).astype(np.uint32)
    labels = labels_from_payload(payload["instance"])
    _l, temp, mat, sky = lite.reduced_numpy(ids, labels, camera, t_s)
    dist = cm._data(payload["distance"]).numpy().reshape(hs, ws)
    dist = np.where(sky, 0.0, np.nan_to_num(dist, posinf=0.0)).astype(np.float32)
    assert temp.dtype == np.float32 and dist.dtype == np.float32, (temp.dtype, dist.dtype)
    paths = labels_to_paths(labels)
    names = sorted(paths.items())
    cam = payload["cam"]
    out = c3.OUT / c3.SIM_DIR / "gbuffer.npz"
    np.savez_compressed(
        out,
        temperature_k=temp,
        material_id=mat,
        sky_mask=sky,
        distance_m=dist,
        ids=ids,
        names=np.array([p.rsplit("/", 1)[-1] for _q, p in names]),
        name_ids=np.array([q for q, _p in names], np.uint32),
        view=np.asarray(cam["cameraViewTransform"], np.float64).reshape(4, 4),
        proj=np.asarray(cam["cameraProjection"], np.float64).reshape(4, 4),
        t_s=np.float64(t_s),
        camera_position=pos,
        camera_look=look,
    )
    print(
        json.dumps(
            {
                "wrote": str(out),
                "shape": [hs, ws],
                "names": [n for _q, n in names],
                "material_ids": {
                    p.rsplit("/", 1)[-1]: int(np.unique(mat[ids == q])[0]) for q, p in names if (ids == q).any()
                },
            }
        ),
        flush=True,
    )
    ctx["app"].close()


if __name__ == "__main__":
    sys.exit(main())
