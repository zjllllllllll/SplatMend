# Viewer provenance

`supersplat/` contains unmodified source from the MIT-licensed PlayCanvas
SuperSplat editor, tag `v2.32.5`, commit
`e060989b202548848eb440a5005cd41a8b26f7db` (2026-08-25).

Source: https://github.com/playcanvas/supersplat/tree/v2.32.5

Only the rendering/selection modules imported by `../../src/main.ts` are built.
The original editor UI, publishing, timeline, video, account and document UI
are not included in the product entry point. The license is preserved at
`supersplat/LICENSE`. Dependencies are pinned separately in `package-lock.json`.

The registered export implementation in `../../src/capture.ts` follows the
user-supplied `supersplat-unified-export-hook.js`: the actual offscreen camera,
OpenGL→OpenCV conversion, RGBA16F weighted camera-Z, NPY [H,W,2], and exact RGBA
PNG encoder. There are no webpage injections or online viewer dependencies.
