# Studio-local changes to the pinned SuperSplat viewer

The bundled viewer is based on SuperSplat 2.32.5. Keep its original MIT license.
These changes affect editor interaction only, not the repository's SHARP /
LingBot numerical pipeline or model sources.

- `src/tools/lasso-selection.ts`: clear the uncommitted polygon when deactivated
  or when pointer input is cancelled; release pointer capture only when owned;
  keep completion of an older asynchronous gesture from clearing a newer one.
  The ordinary mask generation and selection/modifier behavior are unchanged.
- Regression coverage: `studio/web/tests/lasso.test.mjs` runs the actual bundled
  controller against a Node DOM stub. It does not perform browser automation,
  cloud uploads, or image generation.
- `src/editor.ts`: optional `areaSelection: 'through'` decouples lasso/rectangle
  hit testing from the visual `rings` overlay. Studio enables it to select every
  projected Gaussian center inside the region, including occluded layers, using
  the existing GPU intersection path. Visible single-point picking, original
  callers' defaults, protected state bits, deletion and undo are unchanged.
- `studio/web/tests/area-selection.test.mjs` reproduces the old visible-only
  failure and tests both area routes, overlapping layers, modifiers, deletion,
  undo/redo and protected state bits with actual editor/history/state operations.
