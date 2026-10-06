import { getData, type ResultsFile } from "../data";
import { esc, km, pct } from "../util";

export async function render(root: HTMLElement): Promise<void> {
  const r = await getData<ResultsFile>("results.json");
  const fmt = new Intl.NumberFormat("en");
  const label = (id: string) => r.models.find((m) => m.id === id)?.label ?? id;
  const best = r.verdict.find((v) => v.horizon === 60);
  const best120 = r.verdict.find((v) => v.horizon === 120);
  const met = r.criteria.filter((c) => c.met).length;
  const row = (id: string, h: number) => r.prediction?.rows.find((x) => x.model === id && x.horizon_min === h);
  const mlId = r.models.find((m) => m.kind === "ml")?.id ?? "gru";
  const ml60 = row(mlId, 60);
  const dr60 = row("dead_reckoning", 60);

  root.innerHTML = `
  <div class="page">
    ${r.note ? `<div class="callout" role="note"><strong>Note:</strong> ${esc(r.note)}</div>` : ""}
    <section class="hero">
      <span class="pill">Maritime domain awareness · public data</span>
      <h1>Where is that ship going, and is it behaving normally?</h1>
      <p class="lead">
        <strong>ais-sentinel</strong> turns raw ship position broadcasts (AIS) into clean tracks
        with honest uncertainty, forecasts where each vessel will be in the next two hours, and
        flags behaviour an analyst would want to look at. It is built end-to-end on
        ${fmt.format(r.counts.points)} public reports from ${fmt.format(r.counts.vessels)} vessels in
        the ${esc(r.region.name)} (Canada–US border waterways), ${esc(r.period.start)} to ${esc(r.period.end)}.
      </p>
      <div class="actions">
        <a class="btn primary" href="#/tracks">Explore vessel tracks</a>
        <a class="btn" href="#/predict">See predictions vs reality</a>
        <a class="btn" href="#/anomalies">Browse flagged behaviour</a>
      </div>
    </section>

    <section aria-labelledby="pillars">
      <h2 id="pillars">Three things it does</h2>
      <div class="cards">
        <article class="card">
          <h3>1 · Tracking</h3>
          <p>AIS positions are noisy, irregular and sometimes wrong. A Kalman filter and an
          <em>Interacting Multiple Model</em> filter (stationary / cruising / turning) smooth them
          into tracks with error bars and reject impossible fixes.</p>
          <a class="more" href="#/tracks">Raw vs filtered tracks →</a>
        </article>
        <article class="card">
          <h3>2 · Prediction</h3>
          <p>Forecasts at 15, 30, 60 and 120 minutes from physics baselines, a "where did similar
          ships go" route model, and recurrent neural networks, each with a calibrated 90%
          uncertainty ellipse.</p>
          ${
            ml60 && dr60
              ? `<p class="small muted">At 60 min (${esc(r.prediction?.split ?? "")}): ${esc(label(mlId))} ${km(ml60.mean_km)} mean error vs
                 ${km(dr60.mean_km)} for straight-line dead reckoning.</p>`
              : ""
          }
          <a class="more" href="#/predict">Predicted vs actual paths →</a>
        </article>
        <article class="card">
          <h3>3 · Anomaly detection</h3>
          <p>Explainable rules for going dark, impossible jumps, loitering, leaving the usual
          lanes and ship-to-ship meetings, measured on thousands of synthetic anomalies injected
          into real tracks.</p>
          <a class="more" href="#/anomalies">Flagged events with explanations →</a>
        </article>
      </div>
    </section>

    <section aria-labelledby="results-h">
      <h2 id="results-h">Key results (honest version)</h2>
      <div class="cards">
        <div class="card">
          <div class="big">${met}/${r.criteria.length}</div>
          <p class="muted small">pre-registered success criteria met. Targets were set before
          seeing any test result; misses are reported, not hidden.</p>
          <a class="more" href="#/results">All criteria →</a>
        </div>
        ${
          best && best120
            ? `<div class="card">
                <div class="big">${pct(-best120.rel)}</div>
                <p class="muted small">lower mean error than the best non-ML baseline
                (${esc(label(best120.best_baseline))}) at 120 min; ${pct(-best.rel)} at 60 min.
                Negative means ML was worse.</p>
              </div>`
            : ""
        }
        <div class="card">
          <div class="big">${fmt.format(r.counts.voyages)}</div>
          <p class="muted small">voyages segmented from ${fmt.format(r.counts.days)} days of data, split by time
          (train May–Aug, validate Sep, locked test ${esc(r.period.test_start)} – ${esc(r.period.test_end)}).</p>
        </div>
      </div>
    </section>

    <section aria-labelledby="how-h">
      <h2 id="how-h">How it works</h2>
      <ol class="steps">
        <li><strong>Download &amp; clean.</strong> Daily nationwide files (~330 MB each) are
        streamed, filtered to the study area, and cleaned (invalid IDs, sentinel values, duplicates).</li>
        <li><strong>Segment.</strong> Reports are cut into voyages at 30-minute silences and assigned
        to train / validation / test months, with buffer days so no voyage leaks across.</li>
        <li><strong>Track.</strong> Kalman and IMM filters, validated first on simulated ships whose
        true path is known.</li>
        <li><strong>Predict.</strong> Every model is scored on the same samples with
        voyage-level bootstrap confidence intervals.</li>
        <li><strong>Detect.</strong> Rules learn what is normal (coverage, ports, lanes) from training
        data, then flag departures with a plain-language reason.</li>
        <li><strong>Publish.</strong> Small precomputed extracts power this static site.</li>
      </ol>
      <div class="callout small">
        Everything here is reproducible with one command from public data; the code, tests and
        reports are on <a href="https://github.com/jrhughes003/ais-sentinel">GitHub</a>.
        Showcase examples favour commercial vessels; pleasure craft are not featured.
      </div>
    </section>
  </div>`;
}
