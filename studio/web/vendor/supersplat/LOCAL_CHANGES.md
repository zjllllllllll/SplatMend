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
