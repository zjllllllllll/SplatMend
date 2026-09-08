import path from 'node:path';
import { cpSync, mkdirSync } from 'node:fs';
import alias from '@rollup/plugin-alias';
import image from '@rollup/plugin-image';
import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';

export default {
    input: 'src/main.ts',
    output: { dir: 'dist', format: 'esm', sourcemap: false },
    plugins: [
        alias({ entries: { playcanvas: path.resolve('node_modules/playcanvas/build/playcanvas/src/index.js') } }),
        typescript({ tsconfig: './tsconfig.json', noEmitOnError: true }),
        resolve(), image({ dom: false }), json(),
        { name: 'studio-public-files', writeBundle() {
            mkdirSync('dist', { recursive: true });
            cpSync('public', 'dist', { recursive: true });
        } }
    ],
    treeshake: 'smallest'
};
