"""
Theme-neutral design system (works in Streamlit light AND dark mode).

2.1 refresh — calm enterprise look:
* one indigo accent + semantic status colours; surfaces are translucent over the theme background, so the same
  CSS reads well in light and dark mode (contrast of text is left to Streamlit's theme, which is WCAG-AA);
* 4/8 px spacing scale, 14 px radius, a single subtle shadow; no animation except a short hover transition
  (disabled with prefers-reduced-motion);
* visible keyboard focus rings on every interactive element (:focus-visible);
* page headers, KPI cards, empty states, onboarding checklist, metadata chips and the floating help launcher (top right, next to the app menu, so it never covers the chat box).

All 2.0 class names are kept (hero, steps, card, chip, badge, metric, meter, status-line, src-*, trace-line…).
Fonts: Inter / JetBrains Mono when reachable, otherwise the system font stack (offline-safe).
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --accent: #4F46E5; --accent-2: #0EA5E9; --ok: #15A05A; --warn: #C27803; --bad: #D64545; --info: #4F46E5;
  --rule: rgba(128,128,128,.22); --soft: rgba(128,128,128,.06); --soft-2: rgba(128,128,128,.11);
  --tint: rgba(79,70,229,.10); --tint-ok: rgba(21,160,90,.10); --tint-warn: rgba(194,120,3,.11);
  --tint-bad: rgba(214,69,69,.10);
  --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --radius: 14px; --radius-sm: 10px; --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08);
}
html, body, .stApp, .stMarkdown, button, input, textarea, select { font-family: var(--sans); }
.block-container { padding-top: 1.2rem; padding-bottom: 5rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.015em; }
h1 { font-weight: 700; }

/* ---------- keyboard accessibility ---------- */
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, textarea:focus-visible,
[data-baseweb="select"]:focus-within, summary:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px !important; border-radius: 8px; }
.sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }

/* ---------- page header ---------- */
.page-head { display:flex; align-items:flex-start; gap:.85rem; margin: .1rem 0 .9rem; }
.page-icon { flex:0 0 auto; width:44px; height:44px; border-radius:12px; display:grid; place-items:center;
             font-size:1.35rem; background: var(--tint); border:1px solid var(--rule); }
.page-title { font-size:1.55rem; font-weight:700; letter-spacing:-.02em; line-height:1.2; margin:0; }
.page-sub { opacity:.75; font-size:.95rem; margin-top:.2rem; max-width:75ch; }
.crumb { text-transform:uppercase; letter-spacing:.07em; font-size:.68rem; font-weight:600; opacity:.55; }

/* ---------- help panel (what / why / how / after / errors) ---------- */
.help-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr)); gap:.6rem; margin:.2rem 0 .6rem; }
.help-cell { border:1px solid var(--rule); border-radius: var(--radius-sm); padding:.65rem .8rem; background: var(--soft); }
.help-cell h4 { margin:0 0 .25rem; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; opacity:.7; }
.help-cell p, .help-cell li { font-size:.88rem; margin:.1rem 0; }

/* ---------- brand + hero ---------- */
.brand { display:flex; align-items:center; gap:.55rem; margin: .1rem 0 .9rem; }
.logo { width: 34px; height: 34px; border-radius: 10px; display:grid; place-items:center; color:#fff; font-weight:700;
        background: var(--accent); font-size: 1.15rem; }
.brand-title { font-weight: 700; font-size: 1.12rem; } .brand-sub { opacity:.6; font-size:.82rem; }
.hero { border-radius: var(--radius); padding: 1.5rem 1.6rem; margin-bottom: 1.1rem;
        background: linear-gradient(135deg, rgba(79,70,229,.12), rgba(14,165,233,.08)); border: 1px solid var(--rule); }
.hero-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: .3rem; }
.hero-sub { opacity: .82; font-size: 1rem; max-width: 66ch; }
.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:.7rem; margin: 1rem 0 .3rem; }
.step { border:1px solid var(--rule); border-radius: 12px; padding:.8rem .9rem; background: var(--soft); }
.step-n { display:inline-grid; place-items:center; width:1.5rem; height:1.5rem; border-radius:50%; font-size:.8rem;
          font-weight:700; color:#fff; background: var(--accent); margin-right:.4rem; }
.step-t { font-weight:600; } .step-d { opacity:.78; font-size:.86rem; margin-top:.25rem; }
.section-label { text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; font-weight: 600; opacity: .62;
                 margin: .9rem 0 .4rem; }

/* ---------- onboarding checklist ---------- */
.onboard { border:1px solid var(--rule); border-radius: var(--radius); padding: .9rem 1rem; background: var(--soft);
           margin-bottom: 1rem; }
.onboard-title { font-weight:700; margin-bottom:.15rem; } .onboard-sub { opacity:.72; font-size:.86rem; }
.ob-bar { height:6px; border-radius:4px; background: var(--soft-2); overflow:hidden; margin:.55rem 0 .65rem; }
.ob-bar > span { display:block; height:100%; background: var(--ok); border-radius:4px; }
.ob-item { display:flex; gap:.55rem; align-items:flex-start; font-size:.9rem; margin:.3rem 0; }
.ob-check { flex:0 0 auto; width:1.25rem; height:1.25rem; border-radius:50%; display:grid; place-items:center;
            font-size:.72rem; font-weight:700; border:1.5px solid var(--rule); }
.ob-item.done .ob-check { background: var(--ok); border-color: var(--ok); color:#fff; }
.ob-item.done .ob-text { opacity:.6; text-decoration: line-through; }

/* ---------- cards ---------- */
.card { border: 1px solid var(--rule); border-radius: var(--radius); padding: 1rem 1.05rem; background: var(--soft);
        transition: border-color .15s ease; height: 100%; box-shadow: var(--shadow); }
.card:hover { border-color: var(--accent); }
.card-title { font-weight: 600; word-break: break-word; margin-bottom: .2rem; font-size: 1rem; }
.card-sub { opacity:.72; font-size:.85rem; margin-bottom:.55rem; }
.stat-row { display:flex; flex-wrap:wrap; gap:.35rem; }
.stat { background: var(--tint); border-radius: 8px; padding: .15rem .55rem; font-size: .8rem; }
.stat b { font-variant-numeric: tabular-nums; }

/* ---------- KPI tiles ---------- */
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(165px,1fr)); gap:.65rem; margin:.2rem 0 .9rem; }
.kpi { border:1px solid var(--rule); border-radius: 12px; padding:.7rem .85rem; background: var(--soft); }
.kpi-label { font-size:.76rem; opacity:.72; display:flex; gap:.35rem; align-items:center; }
.kpi-val { font-size:1.4rem; font-weight:700; font-variant-numeric: tabular-nums; line-height:1.25; margin-top:.15rem; }
.kpi-note { font-size:.74rem; opacity:.62; margin-top:.1rem; }
.kpi.ok { border-left: 3px solid var(--ok); } .kpi.warn { border-left: 3px solid var(--warn); }
.kpi.bad { border-left: 3px solid var(--bad); } .kpi.info { border-left: 3px solid var(--accent); }

/* ---------- empty state ---------- */
.empty { border:1.5px dashed var(--rule); border-radius: var(--radius); padding: 1.4rem 1.2rem; text-align:center;
         background: var(--soft); margin: .4rem 0 1rem; }
.empty-icon { font-size: 2rem; } .empty-title { font-weight:700; margin:.3rem 0 .2rem; }
.empty-text { opacity:.75; font-size:.9rem; max-width:56ch; margin:0 auto; }

/* ---------- badges, chips, pills ---------- */
.badge { display:inline-flex; align-items:center; gap:.3rem; border-radius:999px; padding:.12rem .65rem; font-size:.78rem;
         font-weight:600; border:1px solid currentColor; margin-right:.35rem; }
.badge.ok { color: var(--ok); } .badge.warn { color: var(--warn); } .badge.bad { color: var(--bad); }
.badge.info { color: var(--accent); }
.badge.cache { color: var(--accent-2); }
.chip { display:inline-block; background: var(--tint); color: var(--accent); border-radius: 7px; padding: .1rem .5rem;
        margin: 0 .3rem .3rem 0; font-size: .78rem; font-weight: 600; }
.chip.user { background: var(--tint-ok); color: var(--ok); }
.chip.auto { background: var(--soft-2); color: inherit; opacity:.85; }
.meta { opacity:.68; font-size:.8rem; }
.status-line { font-size: .86rem; margin: .14rem 0; }
.dot { display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.45rem; vertical-align: 1px; }
.dot.ok { background: var(--ok); } .dot.bad { background: var(--bad); } .dot.warn { background: var(--warn); }
.store { border:1px solid var(--rule); border-radius: 12px; padding:.6rem .75rem; font-size:.82rem; background: var(--soft); }
.conf { display:inline-block; min-width:2.6rem; text-align:center; border-radius:6px; padding:0 .3rem; font-size:.74rem;
        font-weight:600; font-variant-numeric: tabular-nums; }
.conf.hi { background: var(--tint-ok); color: var(--ok); } .conf.mid { background: var(--tint-warn); color: var(--warn); }
.conf.lo { background: var(--tint-bad); color: var(--bad); }

/* ---------- answer summary ---------- */
.ans-sum { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:.45rem; margin:.35rem 0 .5rem; }
.ans-cell { border:1px solid var(--rule); border-radius: 10px; padding:.4rem .6rem; background: var(--soft); }
.ans-k { font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; opacity:.62; }
.ans-v { font-size:.86rem; font-weight:600; word-break: break-word; }

/* ---------- evidence ---------- */
.src-body { font-family: var(--mono); font-size: .78rem; white-space: pre-wrap; word-break: break-word;
            background: var(--soft); border-radius: 8px; padding: .6rem .75rem; }
.src-meta { opacity:.72; font-size:.82rem; margin-bottom:.35rem; }

/* ---------- metrics ---------- */
.metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:.6rem; margin-bottom:.8rem; }
.metric { border:1px solid var(--rule); border-radius:12px; padding:.65rem .8rem; background: var(--soft); }
.metric-name { opacity:.72; font-size:.78rem; } .metric-val { font-size:1.3rem; font-weight:700; font-variant-numeric: tabular-nums; }
.meter { background: var(--soft-2); height:5px; border-radius:3px; margin-top:.35rem; overflow:hidden; }
.meter > span { display:block; height:100%; border-radius:3px; }
.trace-line { font-size:.86rem; margin:.12rem 0; }

/* ---------- FAQ assistant ---------- */
.faq-q { font-weight:600; margin:.2rem 0 .15rem; }
.faq-a { font-size:.9rem; }
.faq-cat { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; opacity:.6; }
div[class*="st-key-faq_fab"] { position: fixed; right: 4.2rem; top: .55rem; z-index: 999990; width: auto !important; }
div[class*="st-key-faq_fab"] button { border-radius: 999px !important; box-shadow: 0 4px 14px rgba(16,24,40,.22);
  font-weight: 600; padding: .45rem 1rem; background: var(--accent) !important; color: #fff !important;
  border: none !important; }
div[class*="st-key-faq_fab"] button:hover { filter: brightness(1.08); }

/* ---------- native widgets ---------- */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 500; }
.stChatMessage { border-radius: var(--radius); }
[data-testid="stChatInput"] textarea { font-size: .98rem; }
[data-testid="stMetric"] { border:1px solid var(--rule); border-radius:12px; padding:.55rem .75rem; background: var(--soft); }
[data-testid="stMetricLabel"] p { font-size:.78rem; opacity:.8; }
[data-testid="stExpander"] details { border-radius: 12px; }
[data-testid="stSidebar"] .section-label { margin-top: .8rem; }
[data-testid="stSidebarNav"] a { border-radius: 8px; }

@media (max-width: 640px) {
  .hero { padding: 1.1rem; } .hero-title { font-size: 1.3rem; } .page-title { font-size: 1.3rem; }
  .page-icon { width:38px; height:38px; font-size:1.15rem; }
  .block-container { padding-left: .8rem; padding-right: .8rem; }
  div[class*="st-key-faq_fab"] { right: 3.4rem; top: .45rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""
