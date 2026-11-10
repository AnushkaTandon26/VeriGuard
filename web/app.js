const input = document.querySelector('#image-input');
const zone = document.querySelector('#drop-zone');
const preview = document.querySelector('#preview');
const emptyUpload = document.querySelector('#empty-upload');
const replaceButton = document.querySelector('#replace-button');
const form = document.querySelector('#analysis-form');
const analyseButton = document.querySelector('#analyse-button');
const message = document.querySelector('#form-message');
const resultCard = document.querySelector('#result-card');
const resultState = document.querySelector('#result-state');
const resultEmpty = document.querySelector('#result-empty');
const resultData = document.querySelector('#result-data');
const imageDetails = document.querySelector('#image-details');
const historyList = document.querySelector('#history-list');
const historyEmpty = document.querySelector('#history-empty');
const clearHistory = document.querySelector('#clear-history');
const sampleList = document.querySelector('#sample-list');
const downloadReport = document.querySelector('#download-report');
const resetScanButton = document.querySelector('#reset-scan');
const confidenceRing = document.querySelector('#confidence-ring');
const pipelineSteps = [...document.querySelectorAll('.pipeline-step')];
const pageViews = [...document.querySelectorAll('[data-page-view]')];
const pageButtons = [...document.querySelectorAll('[data-page]')];
const pageTitles = {
  dashboard: ['SECURITY WORKSPACE', 'Dashboard'],
  image: ['DETECTION STUDIO', 'Image scan'],
  voice: ['DETECTION STUDIO', 'Voice scan'],
  history: ['DETECTION LEDGER', 'Scan history'],
  models: ['MODEL REGISTRY', 'Detection models'],
};

let selectedFile;
let imageUrl;
let lastResult;
const historyStorageKey = 'veriguard-scan-history-v1';
const history = loadHistory();

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function selectPage(page, updateHash = true) {
  const target = pageTitles[page] ? page : 'dashboard';
  pageViews.forEach((view) => view.classList.toggle('is-active', view.dataset.pageView === target));
  pageButtons.forEach((button) => button.classList.toggle('is-active', button.dataset.page === target));
  document.querySelector('#page-eyebrow').textContent = pageTitles[target][0];
  document.querySelector('#page-title').textContent = pageTitles[target][1];
  if (updateHash && window.location.hash !== `#${target}`) window.history.replaceState(null, '', `#${target}`);
  window.scrollTo({ top: 0, behavior: 'auto' });
}

pageButtons.forEach((button) => button.addEventListener('click', () => selectPage(button.dataset.page)));
document.querySelectorAll('[data-page-link]').forEach((link) => link.addEventListener('click', (event) => {
  event.preventDefault();
  selectPage(link.dataset.pageLink);
}));
document.querySelectorAll('a[href="#analyse"], a[href="#voice-analyse"]').forEach((link) => link.addEventListener('click', (event) => {
  event.preventDefault();
  selectPage(link.getAttribute('href') === '#analyse' ? 'image' : 'voice');
}));
window.addEventListener('hashchange', () => selectPage(window.location.hash.slice(1), false));

function setPipelineStage(activeIndex) {
  pipelineSteps.forEach((step, index) => {
    step.classList.toggle('is-complete', index < activeIndex);
    step.classList.toggle('is-current', index === activeIndex);
  });
}

function markResultStale() {
  lastResult = undefined;
  resultCard.className = 'result-card idle';
  resultState.textContent = 'Ready to scan';
  resultEmpty.innerHTML = '<span class="scan-icon">◌</span><p>Image selected. Run the scan to generate<br>an updated authenticity report.</p>';
  resultEmpty.hidden = false;
  resultData.hidden = true;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
}

function showFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    message.textContent = 'Please choose a JPG, PNG, or WebP image.';
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    message.textContent = 'Please choose an image smaller than 12 MB.';
    return;
  }
  selectedFile = file;
  markResultStale();
  setPipelineStage(1);
  message.textContent = file.name;
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = URL.createObjectURL(file);
  preview.src = imageUrl;
  preview.hidden = false;
  emptyUpload.hidden = true;
  replaceButton.hidden = false;
  zone.classList.add('has-image');
  analyseButton.disabled = false;
  imageDetails.hidden = false;
  document.querySelector('#file-type').textContent = file.type.replace('image/', '').toUpperCase();
  document.querySelector('#file-size').textContent = formatSize(file.size);
  const image = new Image();
  image.onload = () => { document.querySelector('#file-resolution').textContent = `${image.naturalWidth} × ${image.naturalHeight}`; };
  image.src = imageUrl;
}

zone.addEventListener('click', (event) => { if (event.target !== replaceButton) input.click(); });
zone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); } });
replaceButton.addEventListener('click', () => input.click());
input.addEventListener('change', () => showFile(input.files[0]));
['dragenter', 'dragover'].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove('dragging'); }));
zone.addEventListener('drop', (event) => showFile(event.dataTransfer.files[0]));

function setLoading() {
  analyseButton.disabled = true;
  analyseButton.innerHTML = 'Analysing <span class="spinner"></span>';
  zone.classList.add('scanning');
  setPipelineStage(2);
  resultCard.className = 'result-card loading';
  resultState.textContent = 'Scanning';
  resultEmpty.innerHTML = '<span class="loader"></span><p>Loading the FA3 model and inspecting image patterns…</p>';
  resultEmpty.hidden = false;
  resultData.hidden = true;
}

function showResult(data) {
  const isFake = data.label === 'fake';
  const margin = Math.abs(data.probabilities.fake - data.probabilities.real);
  resultCard.className = `result-card ${isFake ? 'fake' : 'real'}`;
  setPipelineStage(3);
  resultState.textContent = 'Complete';
  document.querySelector('#result-label').textContent = isFake ? 'Likely manipulated' : 'Likely authentic';
  document.querySelector('#result-detail').textContent = isFake ? 'The model detected patterns associated with a synthetic or altered face.' : 'The model found stronger evidence of a natural, unaltered face.';
  document.querySelector('#confidence-value').textContent = `${data.confidence}%`;
  confidenceRing.style.setProperty('--score', `${data.confidence}%`);
  document.querySelector('#verdict-icon').textContent = isFake ? '!' : '✓';
  document.querySelector('#verdict-level').textContent = `${data.assessment[0].toUpperCase()}${data.assessment.slice(1)} confidence assessment`;
  document.querySelector('#confidence-meter').style.width = `${data.confidence}%`;
  document.querySelector('#margin-value').textContent = `${margin.toFixed(2)} points`;
  document.querySelector('#fake-probability').textContent = `${data.probabilities.fake}%`;
  document.querySelector('#real-probability').textContent = `${data.probabilities.real}%`;
  document.querySelector('#result-image-meta').textContent = `${data.image.format} · ${data.image.width} × ${data.image.height}px`;
  document.querySelector('#inference-time').textContent = `Inference: ${data.inference_ms} ms`;
  document.querySelector('#scan-id').textContent = `Scan ID: ${data.scan_id}`;
  document.querySelector('#guidance-title').textContent = isFake ? 'Pause before you share' : 'Use this as one signal';
  document.querySelector('#guidance-copy').textContent = isFake
    ? 'Check the original source and compare with trusted footage before acting on the image.'
    : 'A real result is not a guarantee. Verify the source when the context matters.';
  resultEmpty.hidden = true;
  resultData.hidden = false;
  addHistory(data, 'image');
}

function loadHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(historyStorageKey) || '[]');
    return Array.isArray(saved)
      ? saved
        .filter((item) => item && ['image', 'audio'].includes(item.type) && ['real', 'fake'].includes(item.label))
        .slice(0, 30)
        .map((item) => ({ ...item, confidence: Number(item.confidence) || 0 }))
      : [];
  } catch (_) {
    return [];
  }
}

function persistHistory() {
  try { localStorage.setItem(historyStorageKey, JSON.stringify(history)); } catch (_) { /* Private browsing may disable storage. */ }
}

function formatHistoryTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? 'Just now' : date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function addHistory(data, type) {
  history.unshift({
    id: data.scan_id,
    name: data.filename || `${type} scan`,
    type,
    label: data.label,
    confidence: Number(data.confidence),
    timestamp: data.generated_at || new Date().toISOString(),
    inferenceMs: data.inference_ms,
  });
  history.splice(30);
  persistHistory();
  renderDashboard();
}

function renderHistory() {
  const hasHistory = history.length > 0;
  historyEmpty.hidden = hasHistory;
  clearHistory.hidden = !hasHistory;
  if (!hasHistory) {
    historyList.innerHTML = '';
    return;
  }
  historyList.innerHTML = history.map((item) => {
    const isFake = item.label === 'fake';
    const mode = item.type === 'audio' ? 'Voice scan' : 'Image scan';
    return `<article class="history-item ${item.type}">
      <span class="history-mode ${item.type}" aria-label="${mode}">${item.type === 'audio' ? '♫' : '◫'}</span>
      <div class="history-file"><strong>${escapeHtml(item.name)}</strong><small>${mode} · ${formatHistoryTime(item.timestamp)} · ${item.inferenceMs || '—'} ms</small></div>
      <div class="history-confidence"><strong>${item.confidence.toFixed(1)}%</strong><span>confidence</span></div>
      <span class="history-tag ${item.label}">${isFake ? 'Synthetic risk' : 'Likely authentic'}</span>
    </article>`;
  }).join('');
}

function renderChart() {
  const chart = document.querySelector('#confidence-chart');
  const empty = document.querySelector('#confidence-chart-empty');
  const chartRange = document.querySelector('#chart-range');
  const scans = history.slice(0, 8).reverse();
  if (!scans.length) {
    chart.hidden = true;
    chart.style.display = 'none';
    empty.hidden = false;
    chartRange.textContent = 'No scans yet';
    return;
  }
  chart.hidden = false;
  chart.style.display = 'block';
  empty.hidden = true;
  chartRange.textContent = `${scans.length} recent ${scans.length === 1 ? 'scan' : 'scans'}`;
  const left = 42;
  const right = 617;
  const baseline = 144;
  const top = 20;
  const points = scans.map((scan, index) => {
    const x = scans.length === 1 ? (left + right) / 2 : left + ((right - left) * index) / (scans.length - 1);
    const y = baseline - ((Math.max(0, Math.min(100, scan.confidence)) / 100) * (baseline - top));
    return { x, y, scan };
  });
  const lineEl = document.querySelector('#chart-line');
  lineEl.setAttribute('points', points.map(({ x, y }) => `${x},${y}`).join(' '));
  lineEl.setAttribute('stroke', '#4f46e5');
  lineEl.setAttribute('stroke-width', '3.5');
  lineEl.setAttribute('fill', 'none');
  lineEl.setAttribute('stroke-linecap', 'round');
  lineEl.setAttribute('stroke-linejoin', 'round');

  const areaEl = document.querySelector('#chart-area');
  areaEl.setAttribute('d', `M ${points[0].x} ${baseline} L ${points.map(({ x, y }) => `${x} ${y}`).join(' L ')} L ${points[points.length - 1].x} ${baseline} Z`);
  areaEl.setAttribute('fill', 'url(#chart-fill)');

  document.querySelector('#chart-points').innerHTML = points.map(({ x, y, scan }) => {
    const color = scan.label === 'fake' ? '#dc2626' : '#059669';
    return `<circle class="chart-point ${scan.label}" cx="${x}" cy="${y}" r="5.5" fill="${color}" stroke="#ffffff" stroke-width="2.5"><title>${escapeHtml(scan.name)}: ${scan.confidence.toFixed(1)}%</title></circle>`;
  }).join('');
}

function renderDashboard() {
  const total = history.length;
  const fake = history.filter((item) => item.label === 'fake').length;
  const real = total - fake;
  const average = total ? history.reduce((sum, item) => sum + item.confidence, 0) / total : 0;
  const fakePercent = total ? (fake / total) * 100 : 0;
  const realPercent = total ? 100 - fakePercent : 0;
  document.querySelector('#scan-total').textContent = total;
  document.querySelector('#scan-fake').textContent = fake;
  document.querySelector('#scan-real').textContent = real;
  document.querySelector('#average-confidence').textContent = total ? `${average.toFixed(1)}%` : '—';
  document.querySelector('#total-scan-note').textContent = total ? 'Stored in this browser' : 'No saved scans yet';
  document.querySelector('#fake-rate').textContent = total ? `${fakePercent.toFixed(0)}% of all scans` : 'No scan data yet';
  document.querySelector('#real-rate').textContent = total ? `${realPercent.toFixed(0)}% of all scans` : 'No scan data yet';
  document.querySelector('#risk-fake-count').textContent = fake;
  document.querySelector('#risk-real-count').textContent = real;
  document.querySelector('#risk-confidence').textContent = total ? `${average.toFixed(1)}%` : '—';
  document.querySelector('#risk-donut-value').textContent = total ? `${fakePercent.toFixed(0)}%` : '—';
  document.querySelector('#risk-donut').style.setProperty('--synthetic-rate', `${fakePercent}%`);
  document.querySelector('#distribution-label').textContent = total ? `${total} saved ${total === 1 ? 'scan' : 'scans'}` : 'Ready for your first scan';
  document.querySelector('#fake-distribution').style.width = `${fakePercent}%`;
  document.querySelector('#real-distribution').style.width = `${realPercent}%`;
  renderChart();
  renderHistory();
}

clearHistory.addEventListener('click', () => {
  history.length = 0;
  persistHistory();
  renderDashboard();
});

function resetScan() {
  selectedFile = undefined;
  input.value = '';
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = undefined;
  preview.removeAttribute('src');
  preview.hidden = true;
  emptyUpload.hidden = false;
  replaceButton.hidden = true;
  zone.classList.remove('has-image', 'scanning');
  imageDetails.hidden = true;
  message.textContent = '';
  analyseButton.disabled = true;
  resultCard.className = 'result-card idle';
  resultState.textContent = 'Awaiting image';
  resultEmpty.innerHTML = '<span class="scan-icon">◌</span><p>Upload an image to generate<br>an authenticity report.</p>';
  resultEmpty.hidden = false;
  resultData.hidden = true;
  setPipelineStage(0);
  document.querySelector('#analyse').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadLastReport() {
  if (!lastResult) return;
  const report = { generated_at: new Date().toISOString(), ...lastResult };
  const downloadUrl = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = `veriface-report-${lastResult.filename.replace(/[^a-z0-9_-]/gi, '_')}.json`;
  link.click();
  URL.revokeObjectURL(downloadUrl);
}

async function loadDemoSamples() {
  try {
    const response = await fetch('/api/samples');
    const data = await response.json();
    if (!data.samples?.length) throw new Error('No bundled demo images were found.');
    sampleList.innerHTML = data.samples.map((sample) => `
      <article class="sample-card">
        <img src="${escapeHtml(sample.url)}" alt="Included face test image">
        <div>
          <strong>${escapeHtml(sample.name)}</strong>
          <button type="button" data-sample-url="${escapeHtml(sample.url)}" data-sample-name="${escapeHtml(sample.name)}">Use this image →</button>
        </div>
      </article>
    `).join('');
    sampleList.querySelectorAll('button[data-sample-url]').forEach((button) => {
      button.addEventListener('click', async () => {
        button.disabled = true;
        button.textContent = 'Preparing image…';
        try {
          const imageResponse = await fetch(button.dataset.sampleUrl);
          const blob = await imageResponse.blob();
          showFile(new File([blob], button.dataset.sampleName, { type: blob.type || 'image/jpeg' }));
          document.querySelector('#analyse').scrollIntoView({ behavior: 'smooth', block: 'start' });
          form.requestSubmit();
        } finally {
          button.disabled = false;
          button.textContent = 'Use this image →';
        }
      });
    });
  } catch (error) {
    sampleList.innerHTML = `<p class="sample-loading">${escapeHtml(error.message)}</p>`;
  }
}

downloadReport.addEventListener('click', downloadLastReport);
resetScanButton.addEventListener('click', resetScan);

async function updateModelStatus() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    const image = data.image || data;
    const audio = data.audio;
    document.querySelector('#model-status').textContent = audio?.dependencies_ready === false
      ? 'Image ready · install audio dependencies'
      : image.model_loaded ? 'Models loaded · ready' : 'Models ready to load';
  } catch (_) {
    document.querySelector('#model-status').textContent = 'Local model status unavailable';
    document.querySelector('#model-indicator').classList.add('offline');
  }
}

selectPage(window.location.hash.slice(1) || 'dashboard', false);
renderDashboard();
updateModelStatus();
loadDemoSamples();

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  setLoading();
  const body = new FormData();
  body.append('image', selectedFile);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    lastResult = data;
    showResult(data);
    updateModelStatus();
  } catch (error) {
    resultCard.className = 'result-card idle';
    resultState.textContent = 'Could not analyse';
    resultEmpty.innerHTML = '<span class="scan-icon">!</span><p></p>';
    resultEmpty.querySelector('p').textContent = error.message;
    resultEmpty.hidden = false;
    resultData.hidden = true;
    setPipelineStage(1);
  } finally {
    zone.classList.remove('scanning');
    analyseButton.disabled = false;
    analyseButton.innerHTML = 'Run authenticity scan <span>→</span>';
  }
});

const audioInput = document.querySelector('#audio-input');
const audioZone = document.querySelector('#audio-drop-zone');
const audioForm = document.querySelector('#audio-analysis-form');
const audioButton = document.querySelector('#audio-analyse-button');
const audioMessage = document.querySelector('#audio-form-message');
const audioPreview = document.querySelector('#audio-preview');
const audioReplaceButton = document.querySelector('#audio-replace-button');
const audioSelected = document.querySelector('#audio-selected');
const audioEmptyUpload = document.querySelector('#audio-empty-upload');
const audioResultCard = document.querySelector('#audio-result-card');
const audioResultState = document.querySelector('#audio-result-state');
const audioResultEmpty = document.querySelector('#audio-result-empty');
const audioResultData = document.querySelector('#audio-result-data');
const audioSampleList = document.querySelector('#audio-sample-list');
const audioExtensions = ['wav', 'mp3', 'ogg', 'webm', 'm4a', 'mp4', 'flac'];
let selectedAudio;
let audioUrl;

function isSupportedAudio(file) {
  const extension = file.name.split('.').pop().toLowerCase();
  return file.type.startsWith('audio/') || audioExtensions.includes(extension);
}

function clearAudioResult() {
  audioResultCard.className = 'audio-result-card idle';
  audioResultState.textContent = 'Ready to scan';
  audioResultEmpty.innerHTML = '<span class="scan-icon">♫</span><p>Audio selected. Run the scan to generate<br>an updated authenticity report.</p>';
  audioResultEmpty.hidden = false;
  audioResultData.hidden = true;
}

function showAudioFile(file) {
  if (!file) return;
  if (!isSupportedAudio(file)) {
    audioMessage.textContent = 'Please choose WAV, MP3, OGG, WebM, M4A, or FLAC audio.';
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    audioMessage.textContent = 'Please choose audio smaller than 12 MB.';
    return;
  }
  selectedAudio = file;
  clearAudioResult();
  audioMessage.textContent = '';
  document.querySelector('#audio-file-name').textContent = file.name;
  document.querySelector('#audio-file-details').textContent = `${file.type || file.name.split('.').pop().toUpperCase()} · ${formatSize(file.size)}`;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(file);
  audioPreview.src = audioUrl;
  audioPreview.hidden = false;
  audioEmptyUpload.hidden = true;
  audioSelected.hidden = false;
  audioReplaceButton.hidden = false;
  audioZone.classList.add('has-audio');
  audioButton.disabled = false;
}

audioZone.addEventListener('click', (event) => { if (event.target !== audioReplaceButton) audioInput.click(); });
audioZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); audioInput.click(); } });
audioReplaceButton.addEventListener('click', () => audioInput.click());
audioInput.addEventListener('change', () => showAudioFile(audioInput.files[0]));
['dragenter', 'dragover'].forEach((name) => audioZone.addEventListener(name, (event) => { event.preventDefault(); audioZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => audioZone.addEventListener(name, (event) => { event.preventDefault(); audioZone.classList.remove('dragging'); }));
audioZone.addEventListener('drop', (event) => showAudioFile(event.dataTransfer.files[0]));

async function loadAudioDemoSamples() {
  try {
    const response = await fetch('/api/audio-samples');
    const data = await response.json();
    if (!data.samples?.length) throw new Error('No bundled voice samples were found.');
    audioSampleList.innerHTML = data.samples.map((sample) => `
      <article class="audio-sample-card">
        <div class="audio-card-head">
          <div class="audio-badge-icon">♫</div>
          <div class="audio-title-wrap">
            <strong>${escapeHtml(sample.name)}</strong>
            <span class="audio-type-tag ${sample.label}">${sample.label === 'fake' ? 'Synthetic Clone' : 'Natural Voice'}</span>
          </div>
        </div>
        <audio controls preload="metadata" src="${escapeHtml(sample.url)}"></audio>
        <button type="button" class="btn-use-audio" data-audio-sample-url="${escapeHtml(sample.url)}" data-audio-sample-name="${escapeHtml(sample.name)}">Use this voice sample →</button>
      </article>
    `).join('');
    audioSampleList.querySelectorAll('button[data-audio-sample-url]').forEach((button) => {
      button.addEventListener('click', async () => {
        button.disabled = true;
        try {
          const audioResponse = await fetch(button.dataset.audioSampleUrl);
          const blob = await audioResponse.blob();
          showAudioFile(new File([blob], button.dataset.audioSampleName, { type: blob.type || 'audio/wav' }));
          document.querySelector('#voice-analyse').scrollIntoView({ behavior: 'smooth', block: 'start' });
          audioForm.requestSubmit();
        } finally {
          button.disabled = false;
        }
      });
    });
  } catch (error) {
    audioSampleList.innerHTML = `<p class="sample-loading">${escapeHtml(error.message)}</p>`;
  }
}

function showAudioResult(data) {
  const isFake = data.label === 'fake';
  const isFixture = Boolean(data.demo_fixture);
  audioResultCard.className = `audio-result-card ${isFake ? 'fake' : 'real'}`;
  audioResultState.textContent = 'Complete';
  document.querySelector('#audio-verdict-icon').textContent = isFake ? '!' : '✓';
  document.querySelector('#audio-result-label').textContent = isFixture ? 'Synthetic test result' : isFake ? 'Likely synthetic voice' : 'Likely human voice';
  document.querySelector('#audio-verdict-level').textContent = `${data.assessment[0].toUpperCase()}${data.assessment.slice(1)} confidence assessment`;
  document.querySelector('#audio-confidence').textContent = `${data.confidence}%`;
  document.querySelector('#audio-result-detail').textContent = isFixture ? data.notice : isFake
    ? 'The model detected voice patterns associated with audio synthesis or manipulation.'
    : 'The model found stronger evidence of a natural voice signal.';
  document.querySelector('#audio-fake-probability').textContent = `${data.probabilities.fake}%`;
  document.querySelector('#audio-confidence-meter').style.width = `${data.probabilities.fake}%`;
  document.querySelector('#audio-file-meta').textContent = `${data.audio.duration_seconds}s · ${data.audio.features}`;
  document.querySelector('#audio-inference-time').textContent = `Inference: ${data.inference_ms} ms`;
  document.querySelector('#audio-scan-id').textContent = `Scan ID: ${data.scan_id}`;
  audioResultEmpty.hidden = true;
  audioResultData.hidden = false;
  addHistory(data, 'audio');
}

audioForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedAudio) return;
  audioButton.disabled = true;
  audioButton.innerHTML = 'Analysing <span class="spinner"></span>';
  audioZone.classList.add('scanning');
  audioResultCard.className = 'audio-result-card loading';
  audioResultState.textContent = 'Scanning';
  audioResultEmpty.innerHTML = '<span class="loader"></span><p>Checking the uploaded voice sample…</p>';
  audioResultEmpty.hidden = false;
  audioResultData.hidden = true;
  const body = new FormData();
  body.append('audio', selectedAudio);
  try {
    const response = await fetch('/api/predict-audio', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Audio analysis failed.');
    showAudioResult(data);
    updateModelStatus();
  } catch (error) {
    audioResultCard.className = 'audio-result-card idle';
    audioResultState.textContent = 'Could not analyse';
    audioResultEmpty.innerHTML = '<span class="scan-icon">!</span><p></p>';
    audioResultEmpty.querySelector('p').textContent = error.message;
    audioResultEmpty.hidden = false;
    audioResultData.hidden = true;
  } finally {
    audioZone.classList.remove('scanning');
    audioButton.disabled = false;
    audioButton.innerHTML = 'Run voice authenticity scan <span>→</span>';
  }
});

loadAudioDemoSamples();
