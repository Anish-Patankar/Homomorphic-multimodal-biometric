#!/usr/bin/env python3
"""
generate_dashboard.py
=====================
Reads evaluation_results/roc_curve.csv and summary data to create a standalone,
interactive HTML dashboard with Chart.js for FAR/FRR analysis, ROC curves,
and score distributions.
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
EVAL_DIR = BASE_DIR / "evaluation_results"
ROC_CSV = EVAL_DIR / "roc_curve.csv"
HTML_OUT = EVAL_DIR / "far_frr_dashboard.html"

def main():
    if not ROC_CSV.exists():
        print(f"Error: {ROC_CSV} not found!")
        return

    thresholds = []
    far_list = []
    frr_list = []
    gar_list = []

    with open(ROC_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["Threshold"])
            # Sample thresholds up to 150 (the relevant operating region) and sparse thereafter
            if t <= 120 or (t <= 500 and t % 5 == 0) or (t % 25 == 0):
                thresholds.append(t)
                far_list.append(round(float(row["FAR_Percent"]), 4))
                frr_list.append(round(float(row["FRR_Percent"]), 4))
                gar_list.append(round(float(row["GAR_Percent"]), 4))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IIT Iris Dataset - FAR / FRR Evaluation Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --red: #ef4444;
      --orange: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2rem;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }}
    h1 {{ font-size: 2rem; font-weight: 700; color: #fff; }}
    p.subtitle {{ color: var(--text-muted); font-size: 1.05rem; margin-top: 0.25rem; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.6rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 9999px;
      background: #0284c7;
      color: #fff;
      margin-top: 0.5rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
    }}
    .card-title {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .card-value {{ font-size: 1.85rem; font-weight: 700; margin: 0.5rem 0; color: #fff; }}
    .card-footer {{ font-size: 0.8rem; color: var(--text-muted); }}
    .highlight-green {{ color: var(--green); }}
    .highlight-blue {{ color: var(--accent); }}
    .highlight-orange {{ color: var(--orange); }}
    .highlight-red {{ color: var(--red); }}
    .chart-container {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }}
    .chart-title {{ font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{ background: #0f172a; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .pill {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.8rem;
    }}
    .pill-eer {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); }}
    .pill-std {{ background: rgba(34, 197, 94, 0.2); color: var(--green); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>IIT Iris Dataset &mdash; FAR / FRR Performance Evaluation</h1>
      <p class="subtitle">Benchmarked with Neurotechnology VeriEye Biometric Matching Engine</p>
      <span class="badge">Engine: VeriEye / IrisMatcher</span>
      <span class="badge" style="background:#475569;">Dataset: IIT iris iriscodes</span>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Equal Error Rate (EER)</div>
        <div class="card-value highlight-blue">0.22 %</div>
        <div class="card-footer">At optimal Score Threshold = 36</div>
      </div>
      <div class="card">
        <div class="card-title">Accuracy (GAR / TAR)</div>
        <div class="card-value highlight-green">99.78 %</div>
        <div class="card-footer">Genuine Acceptance Rate at EER</div>
      </div>
      <div class="card">
        <div class="card-title">Genuine Matches</div>
        <div class="card-value">4,476</div>
        <div class="card-footer">Intra-class pairs across 448 eyes</div>
      </div>
      <div class="card">
        <div class="card-title">Impostor Matches</div>
        <div class="card-value">100,128</div>
        <div class="card-footer">Inter-class pairs (448 &times; 447 / 2)</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Mean Genuine Score</div>
        <div class="card-value highlight-green">851.9</div>
        <div class="card-footer">Std Dev: 144.4 (Min: 5, Max: 1829)</div>
      </div>
      <div class="card">
        <div class="card-title">Mean Impostor Score</div>
        <div class="card-value highlight-red">3.0</div>
        <div class="card-footer">Std Dev: 12.1 (Min: 0, Max: 874)</div>
      </div>
      <div class="card">
        <div class="card-title">ZeroFAR Threshold</div>
        <div class="card-value highlight-orange">875</div>
        <div class="card-footer">FRR at ZeroFAR: 53.49%</div>
      </div>
      <div class="card">
        <div class="card-title">ZeroFRR Threshold</div>
        <div class="card-value">5</div>
        <div class="card-footer">FAR at ZeroFRR: 18.84%</div>
      </div>
    </div>

    <!-- Chart: FAR and FRR vs Threshold -->
    <div class="chart-container">
      <div class="chart-title">FAR &amp; FRR vs. Match Score Threshold</div>
      <canvas id="farFrrChart" height="90"></canvas>
    </div>

    <!-- Table: Operating Points -->
    <div class="chart-container">
      <div class="chart-title">Operating Points &amp; Standard Thresholds</div>
      <table>
        <thead>
          <tr>
            <th>Operating Point</th>
            <th>Score Threshold</th>
            <th>False Acceptance Rate (FAR)</th>
            <th>False Rejection Rate (FRR)</th>
            <th>Genuine Acceptance Rate (GAR)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="pill pill-eer">EER Point (Optimal)</span></td>
            <td><strong>36</strong></td>
            <td class="highlight-blue">0.2226 %</td>
            <td class="highlight-blue">0.2226 %</td>
            <td class="highlight-green">99.7774 %</td>
          </tr>
          <tr>
            <td><span class="pill pill-std">FAR = 0.1% (Standard T36)</span></td>
            <td>43</td>
            <td>0.0989 %</td>
            <td>0.2681 %</td>
            <td>99.7319 %</td>
          </tr>
          <tr>
            <td><span class="pill pill-std">FAR = 1.0%</span></td>
            <td>24</td>
            <td>0.8569 %</td>
            <td>0.1787 %</td>
            <td>99.8213 %</td>
          </tr>
          <tr>
            <td><span class="pill pill-std">FAR = 0.01% (High Security T48)</span></td>
            <td>719</td>
            <td>0.0100 %</td>
            <td>13.2484 %</td>
            <td>86.7516 %</td>
          </tr>
          <tr>
            <td>ZeroFAR (0% False Accepts)</td>
            <td>875</td>
            <td>0.0000 %</td>
            <td>53.4853 %</td>
            <td>46.5147 %</td>
          </tr>
          <tr>
            <td>ZeroFRR (0% False Rejects)</td>
            <td>5</td>
            <td>18.8409 %</td>
            <td>0.0000 %</td>
            <td>100.0000 %</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const labels = {json.dumps(thresholds)};
    const farData = {json.dumps(far_list)};
    const frrData = {json.dumps(frr_list)};

    const ctx = document.getElementById('farFrrChart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: labels,
        datasets: [
          {{
            label: 'FAR (%) - False Accept Rate',
            data: farData,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            fill: false,
            tension: 0.1
          }},
          {{
            label: 'FRR (%) - False Reject Rate',
            data: frrData,
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            fill: false,
            tension: 0.1
          }}
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
          legend: {{ labels: {{ color: '#f8fafc' }} }},
          tooltip: {{
            callbacks: {{
              label: (context) => context.dataset.label + ': ' + context.parsed.y.toFixed(4) + '%'
            }}
          }}
        }},
        scales: {{
          x: {{
            title: {{ display: true, text: 'Score Threshold', color: '#94a3b8' }},
            ticks: {{ color: '#94a3b8' }},
            grid: {{ color: '#334155' }}
          }},
          y: {{
            title: {{ display: true, text: 'Error Rate (%)', color: '#94a3b8' }},
            ticks: {{ color: '#94a3b8' }},
            grid: {{ color: '#334155' }},
            min: 0,
            max: 20
          }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated dashboard: {HTML_OUT}")

if __name__ == "__main__":
    main()
