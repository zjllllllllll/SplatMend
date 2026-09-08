// Read-only visual fixture. No scene import, job endpoints, model calls or GPU.
// Run with `node tests/ui-preview.mjs`; only /, CSS and the real progress module are served.
import {createServer} from 'node:http';
import {readFileSync} from 'node:fs';

const html = readFileSync(new URL('../public/index.html', import.meta.url), 'utf8')
    .replace('<script type="module" src="/main.js"></script>', `<script type="module">
        import {phase, jobProgress, renderProgress} from '/job-progress.mjs';
        const status = new URLSearchParams(location.search).get('state') || 'pipeline';
        const current = status === 'complete' ? phase(9, '本地界面样例：结果已加载。', 'complete') :
            jobProgress({status, stage:3, image_ready:true, message:'本地进度显示样例，不运行任务或模型。'});
        renderProgress(document, current);
        document.getElementById('status').textContent = '仅用于界面检查；所有任务控件已禁用。';
        for (const control of document.querySelectorAll('button, select, textarea')) control.disabled = true;
        document.querySelector('aside').scrollTop = 0;
    </script>`);
const routes = new Map([
    ['/', ['text/html; charset=utf-8', html]],
    ['/studio.css', ['text/css; charset=utf-8', readFileSync(new URL('../public/studio.css', import.meta.url))]],
    ['/job-progress.mjs', ['text/javascript; charset=utf-8', readFileSync(new URL('../src/job-progress.mjs', import.meta.url))]]
]);
const server = createServer((req,res)=>{
    const route=routes.get(new URL(req.url,'http://localhost').pathname);
    if(req.method!=='GET'||!route){res.writeHead(404);res.end();return;}
    res.writeHead(200, {'Content-Type':route[0], 'Cache-Control':'no-store'});
    res.end(route[1]);
});
server.listen(0,'127.0.0.1',()=>console.log(`Read-only UI fixture: http://127.0.0.1:${server.address().port}/`));
