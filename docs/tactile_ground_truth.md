# MuJoCo fingertip contact force output

DexJoCo exposes one force modality: `tactile_gt`. Its net-force fields reduce
MuJoCo contact constraints returned by `mujoco.mj_contactForce`; its spatial
field distributes those same contact forces over a deterministic 16-taxel
layout derived from the fingertip collision geometry.

## Pipeline

```text
mj_step at 500 Hz
  -> mj_contactForce for force-bearing fingertip contacts
  -> contact-frame sign and orientation handling
  -> torque shift from contact point to fingertip origin (r x F)
  -> fingertip body-local force and torque
  -> separate contact normal load and shear; project onto 16 facing taxels
  -> 10 physics samples reduced into one 20 ms control observation
  -> observation["tactile_gt"]
  -> tactile_gt.* and next_tactile_gt.* in Zarr
```

The extractor and projector are read-only. They do not add geoms, constraints,
actuators, or forces, and therefore do not change the MuJoCo trajectory.

## Per-frame schema

Within each hand, fingertip order is `index, middle, ring, thumb`. Single-hand
tasks use `H=1`; bimanual tasks use `H=2` with hand order `right, left`.

| Field | Shape | Dtype | Unit | Control-window reduction |
| --- | --- | --- | --- | --- |
| `fingertip_wrench_local` | `(H, 4, 6)` | `float32` | N, N*m | Mean over all physics substeps |
| `taxel_force_local` | `(H, 4, 16, 3)` | `float32` | N | Mean of per-substep Gaussian projections |
| `fingertip_normal` | `(H, 4)` | `float32` | N | Mean summed normal load |
| `fingertip_force_peak` | `(H, 4)` | `float32` | N | Maximum summed normal load in one substep |
| `fingertip_impulse_local` | `(H, 4, 3)` | `float32` | N*s | Sum of local force times `physics_dt` |
| `contact_fraction` | `(H, 4)` | `float32` | 1 | Fraction of substeps above the contact threshold |
| `sim_time` | `(1,)` | `float64` | s | MuJoCo time at interval end |

The wrench channel order is `[Fx, Fy, Fz, Tx, Ty, Tz]`, expressed in each
fingertip body frame about that body's origin. Reset observations contain zero
interval statistics.

## Spatial taxel projection

New recordings use schema **`dexjoco.tactile_gt.v3`**. Each fingertip has
16 sites: two cylinder rings and two hemispherical cap rings, with four
circumferential sites per ring. The cap rings lie at 45 degrees on each cap.
Positions and axes include the compiled MuJoCo geom pose. IDs remain
`row * 4 + column`; the first and last rows are on the caps.

The three channels are ideal surface-load features in N:
`[tangent_u, tangent_v, normal_inward]`. They are **not pressure in Pa**, an
elastic sensor simulation, or a slip measurement.

For each contact, the extractor retains the actual inward contact normal and
MuJoCo normal load. The projector first separates force into normal and shear
at the contact. Gaussian weights use Euclidean distance to site centers, with
support restricted to sites whose inward normal faces the contact normal
(positive dot product). Weights sum to one within that support. This prevents
load leaking through the finger onto the opposite surface.

Normal load is distributed as a nonnegative scalar. Shear is rotated by the
minimal rotation between the contact normal and each receiving site's normal,
then expressed in the site's tangent axes. Pure compression therefore produces
zero shear, including at the cap poles. Per-contact normal load and shear
magnitude are conserved before multiple contacts superpose. Opposite shear
contributions can cancel, as expected.

These transported features must **not** be rotated back and summed to estimate
net force: transport intentionally changes the represented direction between
curved tangent planes. `fingertip_wrench_local` remains the authoritative exact
net wrench. Its coordinate transform and moment-arm computation are unchanged.

Projection runs after every physics substep, including zero-contact substeps,
and is averaged across the control window. The separate `fingertip_force_peak`
and `contact_fraction` preserve information about brief contacts that the mean
alone would attenuate. Reset statistics represent no elapsed interval, not a
measurement of absence of contact.

### Compatibility and regeneration

The previous v2 projection could encode cap compression as shear. **Do not
relabel old v2 arrays as v3.** Re-run the physical replay and recording to
produce v3. The generic Zarr merger can read v1, v2 or v3 separately but refuses
to merge different schemas. RLDX training conversion now requires v3.

The RLDX single-hand column is `observation.tactile.contact_features`, with
**216** values: the 192 row-major taxel values followed by 24 summary values.
The summary is finger-major, in order index/middle/ring/thumb; each finger has
`[Fx, Fy, Fz, normal, peak, contact_fraction]`. Force values are in N, the last
value is dimensionless. Dataset feature names enumerate the exact order.
Both tactile and mask0 configurations use 216 physics dimensions. Old
192-dimensional checkpoints/configurations and normalization statistics must
not be reused as if this were the same input schema.

## Joining replay tactile to existing videos

Prefer recording **RGB, robot state, actions and tactile together in the same
dynamical replay**, then use `rldx_replay_lerobot.py`. The optional
`rldx_official_tactile_join.py` path keeps original RGB and now checks every
recorded raw/replay state row before publishing data. It checks position and
table height at 1e-4 m, quaternion components at 1e-4 with equivalent signs,
and finger joints at 1e-3 rad. The initial row must still match exactly in
float32. Any failing row rejects the join, with a field and row diagnostic;
there is no automatic fallback that silently attaches mismatched touch.

A receipt records per-field maximum errors and the number of checked rows.
This is an observation-trajectory check, **not proof of identical hidden
physics state**. Source state vectors do not contain full qpos/qvel/solver
state, and fields named object `*_ori_pose` can be cached initial poses rather
than evolving object poses. Task success alone is not evidence of alignment.
Use synchronized replay observations when original-video contact consistency
cannot be established.

The replay tool supports `--data_only=True` for `pinch_tongs` and `click_mouse`;
both use `render_mode="none"` without constructing a renderer. Other tasks still
reject data-only mode explicitly. The iPad restorer now writes the full recorded
tablet and stand poses, including orientation. Lighting is not present in the
recorded proprio vector and is not recovered from it; a restored initial pose
must not be presented as reconstruction of the original RGB scene.

## MuJoCo force handling

`mj_contactForce` returns a contact-frame wrench applied to `geom2`; the
equal-and-opposite wrench applies to `geom1`. The implementation resolves that
sign, rotates the wrench to the world frame, shifts torque to the fingertip
origin, and then rotates it into the fingertip body frame.

Contacts whose `efc_address < 0` are proximity-only and are excluded. The
extractor maps collision geoms through `model.geom_bodyid`, so it does not
depend on optional geom names.

Extraction runs immediately after every `mj_step`. A `mj_forward` must not be
inserted between the physics step and extraction because it would recompute
contact forces at a different state and corrupt peak/impulse accounting.

## Action alignment

Each Zarr row stores a complete transition:

```text
tactile_gt[t]       : force observation before action[t]
action[t]           : action executed during the row
next_tactile_gt[t]  : force observation after action[t]
```

The final action response is therefore retained instead of being dropped at
episode termination. `tactile_gt.sim_time` is the authoritative simulation
clock used to validate timestamps and `control_dt`.

## Zarr merge behavior

Raw episode arrays have a leading time dimension, for example
`fingertip_wrench_local` is `(T, H, 4, 6)` and `taxel_force_local` is
`(T, H, 4, 16, 3)`. New recordings use schema `dexjoco.tactile_gt.v3`. The
converter preserves rank and dtype, requires current and next streams to be
paired, and normalizes datasets to the canonical `[right, left]` hand axis.

Legacy complete v1 datasets remain readable when converted separately. Different schema versions cannot be merged: missing spatial data and changed
projection semantics must not be represented as equivalent or zero contact.

Merged datasets also contain:

- `sensor_present`: distinguishes an episode with no force stream from valid
  zero-contact force data.
- `hand_valid`: distinguishes available hands from zero padding.

Retired `tactile.*` streams are not part of the output contract and
must never be written into a new merged dataset.

## Verification

The test suite checks:

- `geom1/geom2` force sign against MuJoCo constraint force;
- rotated fingertip frames and `r x F` moment arms;
- zero output for proximity-only contacts;
- per-substep peak, impulse, mean, and contact fraction;
- cap/side geometry, facing Gaussian weights, determinism, and separate normal/shear
  conservation, including a real MuJoCo cap contact;
- single-hand and bimanual taxel shapes and ten-substep temporal averaging;
- reset/step lifecycle and exact substep count in every integrated task;
- current/next Zarr alignment and 216D training feature packing;
- rejection of replay drift after an identical initial state and action sequence;
- single-hand, bimanual, and missing-stream merge behavior.

Run the main and converter suites from their respective package directories:

```bash
PYTHONPATH="$PWD:$PWD/../dexjoco-data-converter/src" \
python -m unittest discover -s tests -p 'test_*.py'

PYTHONPATH="$PWD/src" \
python -m unittest discover -s tests -p 'test_*.py'
```
