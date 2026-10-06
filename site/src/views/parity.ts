// Unlisted diagnostic route (#/parity): recompute the Python parity fixtures in the browser.
// The site test asserts the differences are tiny, so browser and Python stay in sync.
import { buildFeatures } from "../ml/features";
import { loadModel, predict } from "../ml/infer";

export async function render(root: HTMLElement): Promise<void> {
  root.innerHTML = `<div class="page"><h1>Model parity check</h1><pre id="parity">running…</pre></div>`;
  const { meta } = await loadModel();
  let maxSeq = 0, maxCtx = 0, maxPosM = 0, maxCovRel = 0;
  for (const f of meta.fixtures) {
    const { seq, ctx } = buildFeatures(f.raw, f, meta.history_min, meta.ref_lat, meta.ref_lon, meta.groups);
    f.seq.flat().forEach((v, i) => (maxSeq = Math.max(maxSeq, Math.abs(v - seq[i]))));
    f.ctx.forEach((v, i) => (maxCtx = Math.max(maxCtx, Math.abs(v - ctx[i]))));
    const p = await predict(f.raw, f);
    p.lat.forEach((la, k) => {
      const dn = (la - f.pred_lat[k]) * 111_320;
      const de = (p.lon[k] - f.pred_lon[k]) * 111_320 * Math.cos((la * Math.PI) / 180);
      maxPosM = Math.max(maxPosM, Math.hypot(de, dn));
      const want = f.pred_cov[k][0][0];
      maxCovRel = Math.max(maxCovRel, Math.abs(p.cov[k][0] - want) / Math.max(want, 1));
    });
  }
  document.querySelector("#parity")!.textContent = JSON.stringify({ n: meta.fixtures.length, maxSeq, maxCtx, maxPosM, maxCovRel });
}
