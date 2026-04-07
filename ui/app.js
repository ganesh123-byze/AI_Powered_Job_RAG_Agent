/* ══════════════════════════════════════════════════════
   TOAST SYSTEM
══════════════════════════════════════════════════════ */
const TOAST_ICONS  = { error:'✕', success:'✓', warning:'!', info:'i' };
const TOAST_TITLES = { error:'Error', success:'Success', warning:'Warning', info:'Info' };

const ERROR_MESSAGES = {
  'Only PDF':               'Please upload a PDF file — other formats are not supported.',
  'too large':              'File is too large. Please upload a resume under 5 MB.',
  'Upload failed':          'Resume upload failed. Make sure the server is running on port 8000.',
  'Resume not found':       'Session expired — please upload your resume again.',
  'Search failed':          'Job search failed. Check your API keys in the .env file.',
  'NetworkError':           'Cannot connect to server. Make sure FastAPI is running:\nuvicorn api.main:app --reload --port 8000',
  'Failed to fetch':        'Cannot reach the server. Open http://127.0.0.1:8000 directly.',
  'Unexpected end of JSON': 'Server returned an empty response. The pipeline may have crashed — check the terminal.',
  '500':                    'Internal server error. Check the terminal running uvicorn for the full error trace.',
  '404':                    'Endpoint not found. Make sure you are on http://127.0.0.1:8000',
  '422':                    'Invalid request data sent to the server.',
  'ADZUNA':                 'Adzuna API error. Check your ADZUNA_APP_ID and ADZUNA_API_KEY in .env',
  'ChromaDB':               'ChromaDB error. Try deleting the data/chroma_db folder and restarting.',
  'PDF':                    'Could not read the PDF. Try a different resume file.',
  'json':                   'Server returned invalid data. Check the terminal for Python errors.',
};

function getFriendlyError(raw) {
  if (!raw) return 'An unexpected error occurred. Check the terminal for details.';
  const lower = raw.toLowerCase();
  for (const [key, msg] of Object.entries(ERROR_MESSAGES)) {
    if (lower.includes(key.toLowerCase())) return msg;
  }
  return raw.length > 140 ? raw.slice(0, 140) + '…' : raw;
}

function showToast(type, message, duration = 5000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <div class="toast-icon">${TOAST_ICONS[type] || 'i'}</div>
    <div class="toast-content">
      <div class="toast-title">${TOAST_TITLES[type] || type}</div>
      <div class="toast-msg">${message}</div>
    </div>
    <button class="toast-close" onclick="dismissToast(this.parentElement)">×</button>
    <div class="toast-progress" style="animation-duration:${duration}ms"></div>
  `;
  container.appendChild(toast);
  setTimeout(() => dismissToast(toast), duration);
  return toast;
}
function dismissToast(toast) {
  if (!toast || toast.classList.contains('removing')) return;
  toast.classList.add('removing');
  setTimeout(() => toast.remove(), 260);
}
const toast = {
  error:   (msg, dur) => showToast('error',   getFriendlyError(msg), dur),
  success: (msg, dur) => showToast('success', msg, dur),
  warning: (msg, dur) => showToast('warning', msg, dur),
  info:    (msg, dur) => showToast('info',    msg, dur),
};


/* ══════════════════════════════════════════════════════
   URL HELPER
══════════════════════════════════════════════════════ */
function ensureUrl(url) {
  if (!url || url.trim() === '' || url === '#') return '#';
  url = url.trim();
  return (url.startsWith('http://') || url.startsWith('https://')) ? url : 'https://' + url;
}


/* ══════════════════════════════════════════════════════
   ANIMATED CANVAS BACKGROUND
══════════════════════════════════════════════════════ */
(function () {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, blobs;
  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    blobs = [
      { x: W*0.12, y: H*0.18, r:420, vx: 0.22, vy: 0.14, hue:'124,111,252', a:0.13 },
      { x: W*0.82, y: H*0.72, r:350, vx:-0.18, vy: 0.20, hue:'56,189,248',  a:0.10 },
      { x: W*0.50, y: H*0.45, r:280, vx: 0.12, vy:-0.22, hue:'52,211,153',  a:0.08 },
    ];
  }
  function frame() {
    ctx.clearRect(0,0,W,H);
    blobs.forEach(b => {
      b.x += b.vx; b.y += b.vy;
      if (b.x < -b.r || b.x > W+b.r) b.vx *= -1;
      if (b.y < -b.r || b.y > H+b.r) b.vy *= -1;
      const g = ctx.createRadialGradient(b.x,b.y,0,b.x,b.y,b.r);
      g.addColorStop(0, `rgba(${b.hue},${b.a})`);
      g.addColorStop(1, `rgba(${b.hue},0)`);
      ctx.beginPath(); ctx.arc(b.x,b.y,b.r,0,Math.PI*2);
      ctx.fillStyle = g; ctx.fill();
    });
    requestAnimationFrame(frame);
  }
  window.addEventListener('resize', resize);
  resize(); frame();
})();


/* ══════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════ */
let sessionId  = null;
let allResults = [];


/* ══════════════════════════════════════════════════════
   ELEMENTS
══════════════════════════════════════════════════════ */
const dropzone   = document.getElementById('dropzone');
const fileInput  = document.getElementById('file-input');
const dzBody     = document.getElementById('dz-body');
const dzStatus   = document.getElementById('dz-status');
const ctaBtn     = document.getElementById('cta-btn');
const ctaLabel   = document.getElementById('cta-label');
const ctaSpinner = document.getElementById('cta-spinner');
const ctaIcon    = document.getElementById('cta-icon');
const logPanel   = document.getElementById('log-panel');
const logBody    = document.getElementById('log-body');


/* ══════════════════════════════════════════════════════
   DRAG & DROP
══════════════════════════════════════════════════════ */
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
document.getElementById('dz-browse').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });


/* ══════════════════════════════════════════════════════
   FILE UPLOAD
══════════════════════════════════════════════════════ */
async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    toast.error('Only PDF files are accepted — please upload a .pdf resume.');
    setDzStatus('Only PDF files accepted', 'err'); return;
  }
  if (file.size > 5 * 1024 * 1024) {
    toast.error('File too large — max 5 MB. Try compressing your PDF.');
    setDzStatus('File too large — max 5 MB', 'err'); return;
  }
  setDzStatus('Uploading…', '');
  toast.info('Uploading resume…', 2500);

  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/upload-resume', { method:'POST', body:fd });
    let data;
    try { data = await res.json(); } catch { throw new Error('Unexpected end of JSON — server may have crashed'); }
    if (!res.ok) throw new Error(data.detail || `Upload failed (HTTP ${res.status})`);
    sessionId = data.session_id;

    dropzone.classList.add('done');
    dzBody.querySelector('.dz-icon').innerHTML = `
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
        <circle cx="11" cy="11" r="9" stroke="#34d399" stroke-width="1.6"/>
        <path d="M7 11l2.5 2.5L15 8" stroke="#34d399" stroke-width="1.6"
          stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`;
    dzBody.querySelector('.dz-primary').textContent  = file.name;
    dzBody.querySelector('.dz-secondary').textContent = `${(file.size/1024).toFixed(0)} KB`;
    setDzStatus('Ready — click Find matching jobs', 'ok');
    ctaBtn.disabled       = false;
    ctaLabel.textContent  = 'Find matching jobs';
    ctaIcon.style.display = 'block';
    toast.success(`Resume uploaded — ${file.name}`, 3000);
  } catch (err) {
    setDzStatus(err.message, 'err');
    toast.error(err.message);
  }
}

function setDzStatus(msg, type) {
  dzStatus.textContent = msg;
  dzStatus.className   = 'dz-status' + (type ? ' ' + type : '');
}


/* ══════════════════════════════════════════════════════
   PIPELINE STEP CONFIG
   Each step has: sidebar step number, label for sidebar,
   and one or more log lines shown when that step runs.
══════════════════════════════════════════════════════ */
const PIPELINE_STEPS = [
  {
    n: 1,
    sidebar: 'Parse resume',
    logs: [
      '📄 Extracting text from PDF…',
      '🔍 Running LLM skill extraction (pass 1)…',
      '✨ Running skill enrichment pass 2…',
      '✅ Resume parsed — skills & roles identified',
    ],
    durationMs: 6000,
  },
  {
    n: 2,
    sidebar: 'Fetch from Adzuna',
    logs: [
      '🌐 Building smart search queries from your skills…',
      '🔗 Querying Adzuna API (page 1)…',
      '🔗 Querying Adzuna API (page 2)…',
      '🧹 Applying experience & relevance filters…',
      '✅ Jobs fetched and stored in ChromaDB',
    ],
    durationMs: 12000,
  },
  {
    n: 3,
    sidebar: 'Embed + store',
    logs: [
      '🧠 Building BM25 keyword index…',
      '📐 Generating vector embeddings for jobs…',
      '🗄️  Upserting to ChromaDB with cosine index…',
      '✅ Hybrid index ready (BM25 + vector)',
    ],
    durationMs: 8000,
  },
  {
    n: 4,
    sidebar: 'Score + rank',
    logs: [
      '🔀 Running hybrid retrieval (BM25 + semantic)…',
      '⚡ Scoring skill overlap across candidates…',
      '📍 Scoring experience alignment…',
      '🗺️  Scoring location & freshness…',
      '🏆 Ranking top 10 matches by overall score…',
      '✅ Top 10 jobs ranked',
    ],
    durationMs: 10000,
  },
  {
    n: 5,
    sidebar: 'Gap analysis',
    logs: [
      '🔎 Identifying missing skills per job…',
      '📚 Fetching learning resources for skill gaps…',
      '✏️  Rewriting resume bullets for top 3 jobs…',
      '✅ Gap analysis & rewrites complete',
    ],
    durationMs: 8000,
  },
];


/* ══════════════════════════════════════════════════════
   SEARCH PIPELINE
══════════════════════════════════════════════════════ */
async function startSearch() {
  if (!sessionId) { toast.warning('Please upload your resume first.'); return; }

  const locs = document.getElementById('locations').value
    .split(',').map(s => s.trim()).filter(Boolean);
  if (!locs.length) { toast.warning('Enter at least one location — e.g. Hyderabad, Bangalore'); return; }
  if (locs.length > 5) { toast.warning('Maximum 5 locations allowed.'); return; }

  setLoading(true);
  logPanel.style.display = 'block';
  logBody.innerHTML = '';
  document.getElementById('results-section').style.display = 'none';
  resetPipeline();
  document.getElementById('log-label').textContent = 'Pipeline running';
  document.getElementById('log-dot').classList.add('pulse');

  toast.info('Pipeline started — this takes 60–90 seconds', 7000);

  // ── Animated step-by-step log display ──────────────────────────────
  let stepIdx = 0;
  let logIdx  = 0;
  let logTimer = null;
  let stepTimer = null;
  let cancelled = false;

  function advanceLogs() {
    if (cancelled) return;
    const step = PIPELINE_STEPS[stepIdx];
    if (!step) return;

    activateStep(step.n);

    // Show one log line at a time
    if (logIdx < step.logs.length) {
      addLog(`[Step ${step.n}/5 · ${step.sidebar}]  ${step.logs[logIdx]}`);
      logIdx++;
      // Schedule next log line within same step
      const delay = (step.durationMs / step.logs.length);
      logTimer = setTimeout(advanceLogs, delay);
    } else {
      // Step done → move to next step
      stepIdx++;
      logIdx = 0;
      if (stepIdx < PIPELINE_STEPS.length) {
        stepTimer = setTimeout(advanceLogs, 400);
      }
    }
  }

  advanceLogs();

  // ── API call ────────────────────────────────────────────────────────
  try {
    const res = await fetch('/search-jobs', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        session_id:       sessionId,
        experience_level: document.getElementById('exp-level').value,
        freshness_days:   parseInt(document.getElementById('freshness').value),
        locations:        locs,
      }),
    });

    // Stop timers
    cancelled = true;
    clearTimeout(logTimer);
    clearTimeout(stepTimer);

    let data;
    try { data = await res.json(); }
    catch { throw new Error('Unexpected end of JSON — check terminal for Python errors'); }

    if (!res.ok) {
      const detail = data?.detail || `Server error (HTTP ${res.status})`;
      throw new Error(detail);
    }

    if (!data.results || data.results.length === 0) {
      toast.warning('No jobs found for your profile and location. Try broader locations or increase freshness days.');
      doneAllSteps();
      addLog('⚠️  No results returned — try broader locations or more freshness days');
      document.getElementById('log-label').textContent = 'Complete';
      document.getElementById('log-dot').classList.remove('pulse');
      setLoading(false);
      return;
    }

    doneAllSteps();
    addLog(`🎉 Pipeline complete — ${data.total_jobs_found || data.results.length} jobs found for ${data.candidate}`);
    document.getElementById('log-label').textContent = 'Complete';
    document.getElementById('log-dot').classList.remove('pulse');

    // ── DEDUPLICATION ──────────────────────────────────────────────────
    // Remove duplicate jobs before storing and rendering.
    // A duplicate is defined as: same (title + company) OR same apply_url OR same job_id.
    const deduped = deduplicateResults(data.results);
    if (deduped.length < data.results.length) {
      const removed = data.results.length - deduped.length;
      addLog(`🧹 Removed ${removed} duplicate job${removed > 1 ? 's' : ''} from results`);
      toast.info(`${removed} duplicate${removed > 1 ? 's' : ''} removed from results`, 3500);
    }

    allResults = deduped;
    renderResults(data, deduped);
    toast.success(`Top ${deduped.length} unique matches found for ${data.candidate}!`, 4000);

  } catch (err) {
    cancelled = true;
    clearTimeout(logTimer);
    clearTimeout(stepTimer);
    addLog('❌ Error: ' + err.message, true);
    toast.error(err.message, 8000);
    document.getElementById('log-label').textContent = 'Failed';
    document.getElementById('log-dot').classList.remove('pulse');
  } finally {
    setLoading(false);
  }
}


/* ══════════════════════════════════════════════════════
   DEDUPLICATION  ← KEY FIX
   Three-signal dedup: job_id, apply_url, title+company
══════════════════════════════════════════════════════ */
function deduplicateResults(results) {
  const seenIds   = new Set();
  const seenUrls  = new Set();
  const seenTitles = new Set();
  const out = [];

  for (const job of results) {
    const titleKey = `${(job.title || '').toLowerCase().trim()}||${(job.company || '').toLowerCase().trim()}`;
    const urlKey   = (job.apply_url || '').trim().toLowerCase().replace(/[?#].*$/, ''); // strip query/hash
    const idKey    = (job.job_id || '').trim();

    const isDuplicate =
      (idKey  && seenIds.has(idKey))   ||
      (urlKey && urlKey !== '#' && seenUrls.has(urlKey)) ||
      (titleKey && seenTitles.has(titleKey));

    if (!isDuplicate) {
      if (idKey)    seenIds.add(idKey);
      if (urlKey && urlKey !== '#') seenUrls.add(urlKey);
      if (titleKey) seenTitles.add(titleKey);
      out.push(job);
    }
  }

  return out;
}


/* ══════════════════════════════════════════════════════
   LOADING STATE
══════════════════════════════════════════════════════ */
function setLoading(on) {
  ctaBtn.disabled          = on;
  ctaLabel.style.display   = on ? 'none'   : 'inline';
  ctaSpinner.style.display = on ? 'block'  : 'none';
  ctaIcon.style.display    = on ? 'none'   : 'block';
  if (!on) ctaLabel.textContent = 'Search again';
}


/* ══════════════════════════════════════════════════════
   LOG HELPERS
══════════════════════════════════════════════════════ */
function addLog(msg, isErr = false) {
  const d = document.createElement('div');
  d.className   = 'log-line' + (isErr ? ' err' : '');
  d.textContent = msg;
  logBody.appendChild(d);
  logBody.scrollTop = logBody.scrollHeight;
}


/* ══════════════════════════════════════════════════════
   PIPELINE STEP INDICATOR (sidebar)
══════════════════════════════════════════════════════ */
function resetPipeline() {
  document.querySelectorAll('.pipe-item').forEach(el => el.classList.remove('active','done'));
}
function activateStep(n) {
  document.querySelectorAll('.pipe-item').forEach(el => {
    const sn = +el.dataset.step;
    el.classList.toggle('done',   sn < n);
    el.classList.toggle('active', sn === n);
    if (sn > n) el.classList.remove('active','done');
  });
}
function doneAllSteps() {
  document.querySelectorAll('.pipe-item').forEach(el => {
    el.classList.remove('active');
    el.classList.add('done');
  });
}


/* ══════════════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════════════ */
function renderResults(data, deduped) {
  document.getElementById('results-title').textContent = `Top ${deduped.length} matches`;
  document.getElementById('results-sub').textContent   = `for ${data.candidate}`;
  document.getElementById('results-section').style.display = 'block';
  buildCards(deduped);
}

function sortBy(key, el) {
  document.querySelectorAll('.sort-chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  const sorted = [...allResults].sort((a, b) => {
    if (key === 'skills') return b.skill_score    - a.skill_score;
    if (key === 'fresh')  return b.freshness_score - a.freshness_score;
    return b.overall_score - a.overall_score;
  });
  buildCards(sorted);
}

function buildCards(list) {
  const container = document.getElementById('cards-list');
  container.innerHTML = '';
  list.forEach((job, i) => {
    const card = createCard(job, i + 1);
    card.style.animationDelay = `${i * 0.055}s`;
    container.appendChild(card);
  });
}


/* ══════════════════════════════════════════════════════
   JOB CARD BUILDER
══════════════════════════════════════════════════════ */
function createCard(job, rank) {
  const card = document.createElement('div');
  card.className = 'job-card';

  const scoreColor = job.overall_score >= 70 ? '#34d399'
    : job.overall_score >= 50 ? '#fbbf24' : '#7c6ffc';

  const R    = 22;
  const C    = 2 * Math.PI * R;
  const dash = ((job.overall_score / 100) * C).toFixed(2);

  const freshBucket = job.freshness_bucket || 'older';
  const freshLabel  = {
    today:      'Today',
    this_week:  'This week',
    this_month: 'This month',
    older:      'Older',
  }[freshBucket] ?? freshBucket;

  const bars = [
    { label:'Skills',     val: job.skill_score,     color:'#34d399' },
    { label:'Experience', val: job.exp_score,        color:'#7c6ffc' },
    { label:'Location',   val: job.location_score,   color:'#38bdf8' },
    { label:'Freshness',  val: job.freshness_score,  color:'#fbbf24' },
  ];

  const barsHTML = bars.map(b => `
    <div class="bar-item">
      <div class="bar-meta">
        <span>${b.label}</span>
        <span>${b.val}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${b.val}%;background:${b.color}"></div>
      </div>
    </div>`).join('');

  const skillsHTML = (job.matched_skills && job.matched_skills.length)
    ? job.matched_skills.map(s => `<span class="skill-chip">${s}</span>`).join('')
    : `<span class="no-skills">No direct skill matches found</span>`;

  const gapsHTML = (job.gaps && job.gaps.length) ? `
    <div class="gap-section">
      <div class="gap-section-title">Skills to learn (${job.gaps.length})</div>
      ${job.gaps.map(g => `
        <div class="gap-row">
          <span class="gap-skill">${g.skill}</span>
          ${g.resource && g.resource !== '#'
            ? `<a class="gap-learn" href="${ensureUrl(g.resource)}" target="_blank" rel="noopener noreferrer">Learn →</a>`
            : `<a class="gap-learn" href="https://www.youtube.com/results?search_query=${encodeURIComponent(g.skill)}+tutorial+beginners" target="_blank" rel="noopener noreferrer">Learn →</a>`
          }
          ${g.project ? `<span class="gap-proj">· ${g.project}</span>` : ''}
        </div>`).join('')}
    </div>` : '';

  const rewritesHTML = (job.rewrites && job.rewrites.length) ? `
    <div class="rewrite-section">
      <div class="rewrite-header" onclick="toggleRewrites(this)">
        <span class="rewrite-title">Resume rewrites for this job (${job.rewrites.length})</span>
        <span class="rewrite-toggle">Show ▼</span>
      </div>
      <div class="rewrite-body" style="display:none">
        ${job.rewrites.map(rw => `
          <div class="rewrite-item">
            <div class="rewrite-skill-tag">${rw.skill_targeted || 'General'}</div>
            <div class="rewrite-before">
              <span class="rewrite-label before-label">Before</span>
              <p class="rewrite-text">${rw.original}</p>
            </div>
            <div class="rewrite-after">
              <span class="rewrite-label after-label">After</span>
              <p class="rewrite-text">${rw.rewritten}</p>
            </div>
          </div>`).join('')}
      </div>
    </div>` : '';

  card.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-rank-label">Rank #${rank}</div>
        <div class="card-title">${job.title}</div>
        <div class="card-company">${job.company}</div>
      </div>
      <div class="score-ring">
        <svg width="60" height="60" viewBox="0 0 60 60">
          <circle cx="30" cy="30" r="${R}" fill="none"
            stroke="rgba(255,255,255,0.05)" stroke-width="5"/>
          <circle cx="30" cy="30" r="${R}" fill="none"
            stroke="${scoreColor}" stroke-width="5"
            stroke-dasharray="${dash} ${C.toFixed(2)}"
            stroke-linecap="round"
            transform="rotate(-90 30 30)"/>
          <text x="30" y="35" text-anchor="middle"
            font-family="Space Grotesk,sans-serif"
            font-size="13" font-weight="700"
            fill="${scoreColor}">${job.overall_score}%</text>
        </svg>
        <div class="ring-sub">match</div>
      </div>
    </div>
    <div class="pill-row">
      <span class="pill loc">${job.location}</span>
      <span class="pill ${freshBucket}">${freshLabel}</span>
      ${job.posted_date ? `<span class="pill">${job.posted_date}</span>` : ''}
    </div>
    <div class="bars-grid">${barsHTML}</div>
    <div class="skills-wrap">${skillsHTML}</div>
    ${gapsHTML}
    ${rewritesHTML}
    <div class="card-footer">
      <a class="apply-btn" href="${ensureUrl(job.apply_url)}" target="_blank" rel="noopener noreferrer">
        Apply now
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
          <path d="M2 6.5h9M7 3l3.5 3.5L7 10"
            stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
            stroke-linejoin="round"/>
        </svg>
      </a>
    </div>`;

  return card;
}

function toggleRewrites(header) {
  const body   = header.nextElementSibling;
  const toggle = header.querySelector('.rewrite-toggle');
  const isOpen = body.style.display !== 'none';
  body.style.display  = isOpen ? 'none' : 'block';
  toggle.textContent  = isOpen ? 'Show ▼' : 'Hide ▲';
}