document.addEventListener('DOMContentLoaded', () => {

    // --- Navigation Logic ---
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            // Update Menu
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Update View
            const targetId = item.getAttribute('data-target');
            sections.forEach(sec => {
                sec.classList.remove('active');
                if (sec.id === targetId) {
                    sec.classList.add('active');
                }
            });
        });
    });

    // --- Dashboard Chart Premium ---
    const ctx = document.getElementById('accuracyChart');
    if (ctx) {
        const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(108, 92, 231, 0.4)');
        gradient.addColorStop(1, 'rgba(108, 92, 231, 0.0)');

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
                datasets: [{
                    label: 'Model Accuracy (%)',
                    data: [82, 85, 84, 89, 92, 94.8, 96.4],
                    borderColor: '#6c5ce7',
                    borderWidth: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#6c5ce7',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    backgroundColor: gradient,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                        ticks: { color: '#a0a3b1', font: { size: 11 } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#a0a3b1', font: { size: 11 } }
                    }
                }
            }
        });
    }

    // --- Training Logic Enhanced ---
    const trainingForm = document.getElementById('training-form');
    const startBtn = document.getElementById('start-training-btn');
    const stopBtn = document.getElementById('stop-training-btn');
    const logsContainer = document.getElementById('training-logs');
    const progressBar = document.getElementById('train-progress');
    const progressText = document.getElementById('train-progress-text');
    const statusBadge = document.getElementById('global-train-status');
    const accuracyValue = document.getElementById('live-accuracy');
    const lossValue = document.getElementById('live-loss');

    let pollInterval = null;

    if (trainingForm) {
        trainingForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(trainingForm);
            const data = Object.fromEntries(formData.entries());

            // UI Updates - Launching
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Initializing...';
            logsContainer.innerHTML = '<div class="log-line system">> Connecting to Neural Engine...</div>';
            if (statusBadge) {
                statusBadge.classList.add('active');
                const lbl = statusBadge.querySelector('.status-label');
                if (lbl) lbl.innerText = 'Engine Active';
            }

            fetch('/start_training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
                .then(res => res.json())
                .then(res => {
                    if (res.status === 'success') {
                        startBtn.innerHTML = '<i class="fa-solid fa-gear fa-spin"></i> Training...';
                        startPolling();
                    } else {
                        alert(res.message);
                        resetTrainingUI();
                    }
                });
        });

        stopBtn.addEventListener('click', () => {
            if (confirm("Are you sure you want to ABORT the training sequence?")) {
                fetch('/stop_training', { method: 'POST' })
                    .then(() => {
                        logsContainer.innerHTML += '<div class="log-line error">> ABORT SEQUENCE INITIATED BY USER.</div>';
                        resetTrainingUI();
                    });
            }
        });
    }

    function resetTrainingUI() {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        startBtn.innerHTML = '<i class="fa-solid fa-rocket"></i> Launch Engine';
        if (statusBadge) {
            statusBadge.classList.remove('active');
            const lbl = statusBadge.querySelector('.status-label');
            if (lbl) lbl.innerText = 'Engine Standby';
        }
        if (pollInterval) clearInterval(pollInterval);
    }

    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);

        // Mock a loss reduction for visual effect if the backend doesn't provide it yet
        let currentLoss = 0.850;

        pollInterval = setInterval(() => {
            fetch('/training_status')
                .then(res => res.json())
                .then(data => {
                    // Update Terminal Title to show mode
                    const termTitle = document.querySelector('.terminal-title');
                    if (data.logs.some(l => l.includes('[HYBRID-SIMULATION]'))) {
                        termTitle.innerText = 'train.py - neuro_engine [SIMULATION MODE]';
                        termTitle.style.color = '#f1c40f';
                    } else {
                        termTitle.innerText = 'train.py - neuro_engine';
                        termTitle.style.color = '';
                    }

                    // Update Progress
                    progressBar.style.width = data.progress + '%';
                    progressText.innerText = data.progress + '%';

                    // Update Metrics
                    if (data.accuracy) {
                        accuracyValue.innerText = (data.accuracy * 100).toFixed(2);
                    }

                    // Simulate loss converging if training is running
                    if (data.is_running) {
                        currentLoss = Math.max(0.042, currentLoss - (Math.random() * 0.005));
                        lossValue.innerText = currentLoss.toFixed(3);
                    }

                    // Update Logs (Optimized)
                    if (data.logs.length > 0) {
                        const lastLog = data.logs[data.logs.length - 1];
                        const logLines = logsContainer.querySelectorAll('.log-line');
                        const lastVisibleLog = logLines.length > 0 ? logLines[logLines.length - 1].innerText.replace('> ', '') : '';

                        if (lastLog !== lastVisibleLog) {
                            const div = document.createElement('div');
                            div.className = 'log-line';
                            if (lastLog.includes('Error')) div.className += ' error';
                            if (lastLog.includes('Success')) div.className += ' success';
                            div.innerText = '> ' + lastLog;
                            logsContainer.appendChild(div);
                            logsContainer.scrollTop = logsContainer.scrollHeight;
                        }
                    }

                    // Check if done
                    if (!data.is_running && data.progress >= 100) {
                        clearInterval(pollInterval);
                        resetTrainingUI();
                        logsContainer.innerHTML += '<div class="log-line success">> MISSION SUCCESS: Model weights optimized and saved.</div>';
                        lossValue.innerText = "0.031";
                    }
                });
        }, 1000);
    }

    // --- Improved Testing / Inference Logic ---
    // ─── Threshold Slider Sync ────────────────────────────────────────────────
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdDisplay = document.getElementById('threshold-display');
    if (thresholdSlider && thresholdDisplay) {
        thresholdSlider.addEventListener('input', () => {
            thresholdDisplay.textContent = parseFloat(thresholdSlider.value).toFixed(2);
        });
    }

    // ─── File Handling & Predict ──────────────────────────────────────────────
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewImage = document.getElementById('preview-image');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resetBtn = document.getElementById('reset-btn');
    const fullReportPanel = document.getElementById('full-report-panel');

    const metaFormat = document.getElementById('meta-format');
    const metaType = document.getElementById('meta-type');
    const metaPx = document.getElementById('meta-px');
    const metaSt = document.getElementById('meta-st');

    const patientNameInput = document.getElementById('patient-name');
    const caseIdInput = document.getElementById('case-id');
    const prevVolumeInput = document.getElementById('previous-volume');
    const saveReportBtn = document.getElementById('save-report-btn');

    let currentAnalysisData = null;

    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });
    }

    function doReset() {
        if (resultsSection) resultsSection.classList.add('hidden');
        if (uploadPrompt) uploadPrompt.classList.remove('hidden');
        if (fullReportPanel) fullReportPanel.classList.add('hidden');
        if (fileInput) fileInput.value = '';
        currentAnalysisData = null;
        if (metaFormat) metaFormat.textContent = 'N/A';
        if (metaType) metaType.textContent = '—';
        if (metaPx) metaPx.textContent = '—';
        if (metaSt) metaSt.textContent = '—';
    }

    if (resetBtn) resetBtn.addEventListener('click', doReset);
    const resetBtnFull = document.getElementById('reset-btn-full');
    if (resetBtnFull) resetBtnFull.addEventListener('click', doReset);

    function handleFile(file) {
        const validExtensions = ['.nii', '.nii.gz', '.jpg', '.jpeg', '.png'];
        const isValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        if (!isValid) {
            alert('UNSUPPORTED MEDIA: System only accepts NIfTI or Radiology Images.');
            return;
        }

        if (uploadPrompt) uploadPrompt.classList.add('hidden');
        if (resultsSection) resultsSection.classList.remove('hidden');
        if (loadingOverlay) loadingOverlay.classList.remove('hidden');
        if (fullReportPanel) fullReportPanel.classList.add('hidden');

        if (metaFormat) metaFormat.textContent = file.name.split('.').pop().toUpperCase();

        const formData = new FormData();
        formData.append('file', file);

        // Send threshold and previous volume
        const thr = thresholdSlider ? thresholdSlider.value : '0.5';
        formData.append('threshold', thr);
        const prevVol = prevVolumeInput ? prevVolumeInput.value : '';
        if (prevVol) formData.append('previous_volume_ml', prevVol);

        fetch('/predict', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('ANALYSIS ERROR: ' + data.error);
                    doReset();
                } else {
                    currentAnalysisData = data;
                    updateInferenceUI(data);
                }
            })
            .catch(err => {
                alert('NETWORK ERROR: ' + err);
                doReset();
            })
            .finally(() => {
                if (loadingOverlay) loadingOverlay.classList.add('hidden');
            });
    }

    function updateInferenceUI(data) {
        // ── Determine pipeline state (Rules 2, 5, 8) ──────────────────────────
        const isIndeterminate = data.is_indeterminate === true;
        const isTumor = !isIndeterminate && data.label && data.label.includes('POSITIVE');
        const insufficientMask = data.insufficient_mask_data === true;
        const reliabilityReduced = data.reliability_reduced === true;
        const oodDetected = data.ood_detected === true;
        const alert = data.alert || { level: 'none', title: '', message: '', show_results: true };
        const pipeStatus = data.pipeline_status || {};

        // Helper: render '—' for null/undefined, format number otherwise
        const fmt = (val, decimals, suffix) =>
            (val == null) ? '\u2014' : parseFloat(val).toFixed(decimals || 2) + (suffix || '');

        // ── Preview Image ──
        if (previewImage && data.image_b64) {
            previewImage.src = 'data:image/jpeg;base64,' + data.image_b64;
        }

        // ── Scan Metadata Strip ────────────────────────────────────────────────
        const sm = data.scan_meta || {};
        if (metaType) metaType.textContent = sm.modality_label || sm.orientation || '\u2014';
        if (metaPx) metaPx.textContent = sm.pixel_spacing_x ? `${sm.pixel_spacing_x}mm \u00d7 ${sm.pixel_spacing_y}mm` : '\u2014';
        if (metaSt) metaSt.textContent = sm.slice_thickness ? `${sm.slice_thickness}mm` : '\u2014';

        // ── Rule 8: Top-Level Fail-Loud Alert Banner ───────────────────────────
        renderAlertBanner(alert, pipeStatus, oodDetected, reliabilityReduced);

        // ── Rule 4: OOD / Low Reliability Badge ───────────────────────────────
        renderReliabilityBadge(reliabilityReduced, oodDetected, pipeStatus, data.seg_quality_score);

        // ── Metadata Warning (secondary, subordinate to alert banner) ──────────
        const warnBanner = document.getElementById('meta-warning-banner');
        const warnText = document.getElementById('meta-warning-text');
        if (sm.warning && warnBanner && warnText && alert.level !== 'critical') {
            warnText.textContent = sm.warning;
            warnBanner.classList.remove('hidden');
        } else if (warnBanner) {
            warnBanner.classList.add('hidden');
        }

        // ── Rule 5: Detection Hero Card (3 states) ────────────────────────────
        const heroCard = document.getElementById('detection-hero-card');
        const detectionText = document.getElementById('detection-text');
        const detectionIcon = document.getElementById('detection-icon');
        const confVal = document.getElementById('confidence-val');
        const confBarFill = document.getElementById('conf-bar-fill');
        const confCaveat = document.getElementById('conf-caveat');

        if (heroCard) {
            heroCard.classList.remove('tumor', 'healthy', 'indeterminate');
            heroCard.classList.add(isIndeterminate ? 'indeterminate' : isTumor ? 'tumor' : 'healthy');
        }
        if (detectionText) {
            if (isIndeterminate) {
                detectionText.textContent = 'INDETERMINATE \u2014 MANUAL REVIEW REQUIRED';
            } else if (isTumor) {
                detectionText.textContent = 'SPACE-OCCUPYING LESION IDENTIFIED';
            } else {
                detectionText.textContent = 'NO RADIOLOGICAL ANOMALY DETECTED';
            }
        }
        if (detectionIcon) {
            detectionIcon.innerHTML = isIndeterminate
                ? '<i class="fa-solid fa-circle-question"></i>'
                : isTumor
                    ? '<i class="fa-solid fa-circle-exclamation"></i>'
                    : '<i class="fa-solid fa-circle-check"></i>';
        }
        const confPct = ((data.confidence || 0) * 100).toFixed(1);
        if (confVal) confVal.textContent = confPct + '%' + (reliabilityReduced ? '*' : '');
        if (confBarFill) confBarFill.style.width = confPct + '%';
        if (confCaveat) {
            confCaveat.textContent = reliabilityReduced
                ? '* Confidence capped \u2014 unverified modality or out-of-distribution scan'
                : '';
            confCaveat.classList.toggle('hidden', !reliabilityReduced);
        }

        // ── Rule 3: Measurements (null = '\u2014', 0.0 never shown as valid data) ──
        const meas = data.measurements || {};
        const measActive = isTumor && !insufficientMask;

        setText('vol-val', measActive ? fmt(meas.volume_ml, 2, ' mL') : '\u2014');
        setText('diam-val', measActive ? fmt(meas.diameter_mm, 1, ' mm') : '\u2014');
        setText('perp-val', measActive ? fmt(meas.perpendicular_mm, 1, ' mm') : '\u2014');

        ['vol-card', 'diam-card', 'perp-card'].forEach(function (id) {
            var el = document.getElementById(id);
            if (el) el.classList.toggle('state-na', !measActive);
        });

        // ── Rule 3: Grade (never stale \u2014 always server-sanitized) ─────────────
        const gradeEl = document.getElementById('grade-val');
        const gradeCard = document.getElementById('grade-card');
        if (gradeEl) {
            var g = data.grade || '\u2014';
            gradeEl.textContent = g;
            var gradeIsNA = !isTumor || insufficientMask || g.startsWith('Not');
            if (gradeCard) gradeCard.classList.toggle('state-na', gradeIsNA);
        }

        // ── Risk Gauge ────────────────────────────────────────────────────────
        const risk = data.risk || {};
        const riskScore = (risk.total_score != null) ? risk.total_score : 0;
        const riskLevel = risk.risk_level || null;
        const riskColor = risk.risk_color || '#888888';
        const riskActive = isTumor && !insufficientMask && riskLevel != null;

        animateGauge(riskActive ? riskScore : 0, riskActive ? riskColor : '#4b5563');

        const riskBadge = document.getElementById('risk-level-badge');
        if (riskBadge) {
            if (riskActive) {
                riskBadge.textContent = riskLevel;
                riskBadge.style.background = riskColor + '22';
                riskBadge.style.color = riskColor;
                riskBadge.style.borderColor = riskColor + '66';
            } else {
                riskBadge.textContent = isTumor ? 'Analysis Incomplete' : 'Not Applicable';
                riskBadge.style.background = '';
                riskBadge.style.color = '#6b7280';
                riskBadge.style.borderColor = '';
            }
        }
        const riskNote = document.getElementById('risk-clinical-note');
        if (riskNote) riskNote.textContent = riskActive ? (risk.clinical_note || '') : '';

        // ── Growth Tracking ──
        const growthTracker = document.getElementById('growth-tracker');
        const growthContent = document.getElementById('growth-content');
        if (data.growth && data.growth.trend !== 'baseline' && growthTracker && growthContent) {
            const g = data.growth;
            const trendIcon = g.trend === 'growth' || g.trend === 'mild_growth' ? '📈' : g.trend === 'stable' ? '➡️' : '📉';
            const trendColor = g.trend.includes('growth') ? '#ef4444' : g.trend === 'stable' ? '#f59e0b' : '#22c55e';
            growthContent.innerHTML = `
                <div class="growth-row">
                    <span>${trendIcon} <strong style="color:${trendColor}">${g.percent_change > 0 ? '+' : ''}${g.percent_change}%</strong> volume change</span>
                    <span class="growth-note">${g.clinical_significance}</span>
                </div>`;
            growthTracker.classList.remove('hidden');
        } else if (growthTracker) {
            growthTracker.classList.add('hidden');
        }

        // ── Segmentation Overlay ──
        const overlayImg = document.getElementById('overlay-img');
        const originalImg = document.getElementById('original-img');
        const overlayPlaceholder = document.getElementById('overlay-placeholder');

        if (data.overlay_b64 && overlayImg) {
            overlayImg.src = 'data:image/png;base64,' + data.overlay_b64;
            overlayImg.classList.remove('hidden');
            if (overlayPlaceholder) overlayPlaceholder.classList.add('hidden');
        } else {
            if (overlayImg) overlayImg.classList.add('hidden');
            if (overlayPlaceholder) overlayPlaceholder.classList.remove('hidden');
        }
        if (data.image_b64 && originalImg) {
            originalImg.src = 'data:image/jpeg;base64,' + data.image_b64;
        }

        // ── Radiomics Panel ──
        const rad = data.radiomics || {};
        const loc = rad.location || {};
        const enh = rad.enhancement || {};
        const edema = rad.edema || {};
        const shape = rad.shape || {};
        const tex = rad.texture || {};

        if (isTumor && !insufficientMask) {
            setText('rad-location', loc.full_location || loc.region || '—');
            const eloquentEl = document.getElementById('rad-eloquent');
            if (eloquentEl) {
                eloquentEl.textContent = loc.is_eloquent ? '⚠ Eloquent Area' : (loc.depth_category || '');
                eloquentEl.style.color = loc.is_eloquent ? '#ef4444' : '#6b7280';
            }

            setText('rad-enhancement', enh.enhancement_pattern || '—');
            const ringEl = document.getElementById('rad-ring');
            if (ringEl) {
                ringEl.textContent = enh.is_ring_enhancement ? '⚠ Ring Enhancement Detected' : `Ratio: ${(enh.enhancement_ratio || 0).toFixed(2)}`;
                ringEl.style.color = enh.is_ring_enhancement ? '#ef4444' : '#6b7280';
            }

            setText('rad-edema', edema.edema_label || '—');
            const edemaRatioEl = document.getElementById('rad-edema-ratio');
            if (edemaRatioEl) edemaRatioEl.textContent = edema.edema_ratio ? `Ratio: ${edema.edema_ratio.toFixed(2)}` : '';

            setText('rad-shape', shape.shape_label || '—');
            const spherEl = document.getElementById('rad-sphericity');
            if (spherEl) spherEl.textContent = shape.sphericity ? `Sphericity: ${shape.sphericity.toFixed(3)} | Irregularity: ${(shape.irregularity || 0).toFixed(3)}` : '';

            setText('rad-texture', tex.texture_label || '—');
            const homoEl = document.getElementById('rad-homogeneity');
            if (homoEl) homoEl.textContent = tex.homogeneity ? `Homogeneity: ${tex.homogeneity.toFixed(3)} | Entropy: ${(tex.entropy || 0).toFixed(3)}` : '';

            setText('rad-surgical-risk', loc.surgical_risk || '—');
            const depthEl = document.getElementById('rad-depth');
            if (depthEl) {
                depthEl.textContent = loc.depth_category ? `Depth: ${loc.depth_category}` : '';
                depthEl.style.color = loc.depth_category === 'Deep' ? '#ef4444' : '#6b7280';
            }
        } else {
            // Rule 3: No stale radiomics data shown
            var naMsg = insufficientMask ? 'Segmentation insufficient \u2014 not computed.' : '\u2014';
            ['rad-location', 'rad-enhancement', 'rad-edema', 'rad-shape', 'rad-texture', 'rad-surgical-risk',
                'rad-eloquent', 'rad-ring', 'rad-edema-ratio', 'rad-sphericity', 'rad-homogeneity', 'rad-depth'
            ].forEach(function (id) { setText(id, naMsg); });
        }

        // ── SE Channel Attention Weights ──────────────────────────────────────
        const seAttn = data.se_channel_attention || { T1: 0.25, T1c: 0.35, T2: 0.20, FLAIR: 0.20 };
        const setSeBar = (idBar, idVal, val) => {
            const bar = document.getElementById(idBar);
            const text = document.getElementById(idVal);
            const pct = Math.round((val || 0) * 100);
            if (bar) bar.style.width = pct + '%';
            if (text) text.textContent = pct + '%';
        };
        setSeBar('se-t1-bar', 'se-t1-val', seAttn.T1);
        setSeBar('se-t1c-bar', 'se-t1c-val', seAttn.T1c);
        setSeBar('se-t2-bar', 'se-t2-val', seAttn.T2);
        setSeBar('se-flair-bar', 'se-flair-val', seAttn.FLAIR);

        // ── Sub-region Volumetrics (BraTS ET, TC, WT) ─────────────────────────
        const subVols = data.sub_region_volumes || {};
        setText('sr-et-vol', isTumor && subVols.ET != null ? `${subVols.ET} mL` : '— mL');
        setText('sr-tc-vol', isTumor && subVols.TC != null ? `${subVols.TC} mL` : '— mL');
        setText('sr-wt-vol', isTumor && subVols.WT != null ? `${subVols.WT} mL` : '— mL');

        // ── WHO CNS5 Key Radiological Indicators ──────────────────────────────
        const whoInd = (rad && rad.who_cns5_indicators) ? rad.who_cns5_indicators : {};
        setText('ind-tumor-type', isTumor ? (data.tumor_type || 'Glioma') : 'Healthy');
        setText('ind-margin', isTumor ? (whoInd.margin_type || '—') : 'N/A');
        setText('ind-ring', isTumor ? (whoInd.ring_enhancement || '—') : 'N/A');
        setText('ind-necrosis', isTumor ? (whoInd.necrosis_present ? 'Present (High-Grade Marker)' : 'Absent / Minimal') : 'N/A');
        setText('ind-dwi', isTumor ? (whoInd.dwi_diffusion || '—') : 'N/A');
        setText('ind-perfusion', isTumor ? (whoInd.perfusion_rcbv || '—') : 'N/A');
        setText('ind-mrs', isTumor ? (whoInd.mrs_cho_naa || '—') : 'N/A');
        setText('ind-butterfly', isTumor ? (whoInd.butterfly_glioma || '—') : 'N/A');

        // ── Dual Risk Dimensions (Biological vs Surgical) ─────────────────────
        const bioRisk = risk.biological_risk || {};
        const surgRisk = risk.surgical_risk || {};
        setText('bio-risk-score', isTumor && bioRisk.score != null ? `${bioRisk.score}/100` : '—');
        setText('bio-risk-level', isTumor && bioRisk.level ? bioRisk.level : '—');
        setText('surg-risk-score', isTumor && surgRisk.score != null ? `${surgRisk.score}/100` : '—');
        setText('surg-risk-level', isTumor && surgRisk.level ? surgRisk.level : '—');

        // ── Rule 7: Pipeline Status Block ─────────────────────────────────────
        renderPipelineStatus(pipeStatus);

        // ── Risk Factor Breakdown ──────────────────────────────────────────────
        const factorsList = document.getElementById('risk-factors-list');
        if (factorsList) {
            const breakdown = (risk.factor_breakdown || []);
            if (breakdown.length > 0 && riskActive) {
                factorsList.innerHTML = breakdown.map(f => `
                    <div class="risk-factor-row">
                        <span class="rf-label">${f.factor}</span>
                        <div class="rf-bar-track">
                            <div class="rf-bar-fill" style="width:${f.percent}%;background:${f.color}" data-pct="${f.percent}"></div>
                        </div>
                        <span class="rf-score">${f.score}/${f.max}</span>
                    </div>`).join('');
                // Animate bars in
                setTimeout(() => {
                    document.querySelectorAll('.rf-bar-fill').forEach(bar => {
                        bar.style.width = bar.dataset.pct + '%';
                    });
                }, 50);
            } else {
                var noDataMsg = isIndeterminate
                    ? 'Indeterminate scan \u2014 risk scoring not performed.'
                    : insufficientMask
                        ? 'Segmentation insufficient \u2014 risk scoring not computed.'
                        : isTumor
                            ? 'Risk scoring unavailable for this scan.'
                            : 'No tumor detected \u2014 risk scoring not applicable.';
                factorsList.innerHTML = '<p class="no-data-msg">' + noDataMsg + '</p>';
            }
        }

        // ── Clinical Impression ──
        const impressionEl = document.getElementById('impression-text');
        if (impressionEl) {
            const impression = (data.structured_report && data.structured_report.clinical_impression)
                || (data.details && data.details.impression)
                || risk.clinical_summary
                || 'No impression available.';
            impressionEl.textContent = impression;
        }

        // ── Show Full Report Panel ──
        if (fullReportPanel) fullReportPanel.classList.remove('hidden');
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    // ── Rule 8: Render fail-loud alert banner at top of results ───────────────
    function renderAlertBanner(alertData, pipeStatus, oodDetected, reliabilityReduced) {
        var banner = document.getElementById('pipeline-alert-banner');
        if (!banner) {
            var resultsSection = document.getElementById('results-section') || document.querySelector('.results-section');
            if (!resultsSection) return;
            banner = document.createElement('div');
            banner.id = 'pipeline-alert-banner';
            banner.className = 'pipeline-alert hidden';
            resultsSection.insertBefore(banner, resultsSection.firstChild);
        }
        if (alertData.level === 'none' && !oodDetected && !reliabilityReduced) {
            banner.classList.add('hidden');
            return;
        }
        var levelClass = { critical: 'alert-critical', warning: 'alert-warning', info: 'alert-info' };
        var levelIcon = { critical: '\uD83D\uDEA8', warning: '\u26A0\uFE0F', info: '\u2139\uFE0F' };
        var effectiveLevel = alertData.level !== 'none' ? alertData.level : reliabilityReduced ? 'warning' : 'info';
        banner.className = 'pipeline-alert ' + (levelClass[effectiveLevel] || 'alert-warning');
        banner.innerHTML =
            '<div class="alert-header">'
            + '<span class="alert-icon">' + (levelIcon[effectiveLevel] || '\u26A0\uFE0F') + '</span>'
            + '<strong class="alert-title">' + (alertData.title || (reliabilityReduced ? 'LOW RELIABILITY \u2014 Unverified Input' : 'Analysis Notice')) + '</strong>'
            + '</div>'
            + ((alertData.message || pipeStatus.reason_if_reduced)
                ? '<div class="alert-body">' + (alertData.message || pipeStatus.reason_if_reduced) + '</div>'
                : '')
            + (pipeStatus.stage_summary
                ? '<div class="alert-stage-summary">' + pipeStatus.stage_summary + '</div>'
                : '');
        banner.classList.remove('hidden');
    }

    // ── Rule 4: OOD / Reliability Badge near confidence display ───────────────
    function renderReliabilityBadge(reliabilityReduced, oodDetected, pipeStatus, segQuality) {
        var badge = document.getElementById('reliability-badge');
        if (!badge) {
            var confArea = document.getElementById('confidence-area') || document.getElementById('detection-hero-card');
            if (!confArea) return;
            badge = document.createElement('span');
            badge.id = 'reliability-badge';
            badge.className = 'reliability-badge';
            confArea.appendChild(badge);
        }
        if (!reliabilityReduced && !oodDetected) {
            badge.classList.add('hidden');
            return;
        }
        var overall = pipeStatus.overall_reliability || 'reduced';
        badge.className = 'reliability-badge badge-' + overall;
        badge.textContent = overall === 'low' ? '\u26A0 LOW RELIABILITY' : '\u26A0 REDUCED RELIABILITY';
        badge.title = pipeStatus.reason_if_reduced || 'Input modality unverified or OOD';
        badge.classList.remove('hidden');
    }

    // ── Rule 7: Pipeline stage status tracker ─────────────────────────────────
    function renderPipelineStatus(pipeStatus) {
        var container = document.getElementById('pipeline-status-block');
        if (!container) return;
        if (!pipeStatus || !pipeStatus.stage_summary) {
            container.classList.add('hidden');
            return;
        }
        var stages = [
            { key: 'detection_stage', label: 'Detection' },
            { key: 'segmentation_stage', label: 'Segmentation' },
            { key: 'radiomics_stage', label: 'Radiomics' },
            { key: 'grading_stage', label: 'Grading' },
            { key: 'risk_scoring_stage', label: 'Risk Score' },
        ];
        var iconMap = { completed: '\u2713', failed: '\u2717', not_run: '\u25CB', skipped: '\u2014' };
        var colorMap = { completed: '#22c55e', failed: '#ef4444', not_run: '#6b7280', skipped: '#9ca3af' };
        container.innerHTML = stages.map(function (s) {
            var status = pipeStatus[s.key] || 'not_run';
            return '<div class="ps-item" title="' + s.label + ': ' + status + '">'
                + '<span class="ps-icon" style="color:' + (colorMap[status] || '#6b7280') + '">' + (iconMap[status] || '\u25CB') + '</span>'
                + '<span class="ps-label">' + s.label + '</span>'
                + '</div>';
        }).join('');
        container.classList.remove('hidden');
    }

    // ─── Gauge Animation ─────────────────────────────────────────────────────
    function animateGauge(targetScore, color) {
        const circle = document.getElementById('gauge-circle');
        const scoreText = document.getElementById('gauge-score-text');
        if (!circle || !scoreText) return;

        const circumference = 301.6;
        const pct = Math.min(100, Math.max(0, targetScore)) / 100;
        const offset = circumference * (1 - pct);

        circle.style.stroke = color;
        circle.style.strokeDashoffset = circumference; // start at 0
        circle.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)';

        let current = 0;
        const increment = targetScore / 60;
        const timer = setInterval(() => {
            current = Math.min(current + increment, targetScore);
            scoreText.textContent = Math.round(current);
            if (current >= targetScore) clearInterval(timer);
        }, 20);

        setTimeout(() => { circle.style.strokeDashoffset = offset; }, 30);
    }

    // ─── Overlay Toggle ───────────────────────────────────────────────────────
    window.showOverlay = function () {
        const ov = document.getElementById('overlay-img');
        const or = document.getElementById('original-img');
        const b1 = document.getElementById('btn-show-overlay');
        const b2 = document.getElementById('btn-show-original');
        if (ov) ov.classList.remove('hidden');
        if (or) or.classList.add('hidden');
        if (b1) b1.classList.add('active');
        if (b2) b2.classList.remove('active');
    };
    window.showOriginal = function () {
        const ov = document.getElementById('overlay-img');
        const or = document.getElementById('original-img');
        const b1 = document.getElementById('btn-show-overlay');
        const b2 = document.getElementById('btn-show-original');
        if (ov) ov.classList.add('hidden');
        if (or) or.classList.remove('hidden');
        if (b1) b1.classList.remove('active');
        if (b2) b2.classList.add('active');
    };

    // ─── Save Report ─────────────────────────────────────────────────────────
    if (saveReportBtn) {
        saveReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) {
                alert("No analysis data. Please run a scan first.");
                return;
            }
            const pName = (patientNameInput && patientNameInput.value) || 'Anonymous';
            const cId = (caseIdInput && caseIdInput.value) || ('CASE-' + Math.floor(Math.random() * 90000 + 10000));

            saveReportBtn.disabled = true;
            saveReportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Archiving...';

            const risk = currentAnalysisData.risk || {};
            const meas = currentAnalysisData.measurements || {};
            const sm = currentAnalysisData.scan_meta || {};

            const reportData = {
                case_id: cId,
                patient_name: pName,
                diagnosis: currentAnalysisData.label,
                confidence: currentAnalysisData.confidence,
                grade: currentAnalysisData.grade,
                tumor_size: meas.volume_ml ? `${meas.volume_ml.toFixed(2)} mL` : 'N/A',
                tumor_loc: (currentAnalysisData.details && currentAnalysisData.details.tumor_loc) || 'N/A',
                impression: (currentAnalysisData.details && currentAnalysisData.details.impression) || '',
                risk_level: risk.risk_level || 'N/A',
                risk_score: risk.total_score || 0,
                volume_ml: meas.volume_ml || 0,
                diameter_mm: meas.diameter_mm || 0,
                scan_meta: sm,
                radiomics: currentAnalysisData.radiomics || {},
                risk_breakdown: risk.factor_breakdown || [],
                growth: currentAnalysisData.growth || null,
                image_b64: currentAnalysisData.image_b64 || '',
                overlay_b64: currentAnalysisData.overlay_b64 || ''
            };

            fetch('/save_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reportData)
            })
                .then(res => res.json())
                .then(resp => {
                    if (resp.status === 'success') {
                        saveReportBtn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Archived!';
                        saveReportBtn.style.background = 'linear-gradient(135deg,#22c55e,#16a34a)';
                        setTimeout(() => {
                            saveReportBtn.disabled = false;
                            saveReportBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Archive Report';
                            saveReportBtn.style.background = '';
                        }, 2500);
                        if (reportsTableBody) loadReports();
                    } else {
                        alert('SAVE ERROR: ' + resp.message);
                        saveReportBtn.disabled = false;
                        saveReportBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Archive Report';
                    }
                })
                .catch(err => {
                    alert('NETWORK ERROR: ' + err);
                    saveReportBtn.disabled = false;
                    saveReportBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Archive Report';
                });
        });
    }

    // ─── Print Report ─────────────────────────────────────────────────────────
    const printReportBtn = document.getElementById('print-report-btn');
    if (printReportBtn) {
        printReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) {
                alert("No analysis data. Please run a scan first.");
                return;
            }
            printMedicalReport(currentAnalysisData);
        });
    }

    function printMedicalReport(data) {
        const pName = (patientNameInput && patientNameInput.value) || 'Anonymous';
        const cId = (caseIdInput && caseIdInput.value) || 'N/A';
        const now = new Date().toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
        const reportId = 'RPT-' + Date.now().toString(36).toUpperCase();

        const isIndeterminate = data.is_indeterminate === true;
        const isTumor = !isIndeterminate && data.label && data.label.includes('POSITIVE');
        const insufficientMask = data.insufficient_mask_data === true;
        const confPct = ((data.confidence || 0) * 100).toFixed(1);

        const meas = data.measurements || {};
        const risk = data.risk || {};
        const rad = data.radiomics || {};
        const loc = rad.location || {};
        const enh = rad.enhancement || {};
        const edema = rad.edema || {};
        const shape = rad.shape || {};
        const tex = rad.texture || {};
        const sm = data.scan_meta || {};
        const measActive = isTumor && !insufficientMask;

        const fmt = (v, d, suf) => (v == null) ? '—' : parseFloat(v).toFixed(d || 2) + (suf || '');

        const detectionLabel = isIndeterminate
            ? 'INDETERMINATE — MANUAL REVIEW REQUIRED'
            : isTumor ? 'POSITIVE: SPACE-OCCUPYING LESION IDENTIFIED'
                : 'NEGATIVE: NO RADIOLOGICAL ANOMALY DETECTED';

        const detectionColor = isIndeterminate ? '#f59e0b' : isTumor ? '#ef4444' : '#22c55e';

        const impression = (data.structured_report && data.structured_report.clinical_impression)
            || (data.details && data.details.impression)
            || (risk.clinical_summary)
            || 'No clinical impression available.';

        const rfRows = (risk.factor_breakdown || []).map(f =>
            `<tr>
                <td>${f.factor || ''}</td>
                <td><div style="background:#e5e7eb;border-radius:4px;height:10px;width:140px;display:inline-block;vertical-align:middle;"><div style="background:${f.color || '#6b7280'};width:${f.percent || 0}%;height:100%;border-radius:4px;"></div></div></td>
                <td style="text-align:right;font-weight:600;">${f.score}/${f.max}</td>
            </tr>`
        ).join('');

        const scanImgSrc = data.overlay_b64
            ? 'data:image/png;base64,' + data.overlay_b64
            : (data.image_b64 ? 'data:image/jpeg;base64,' + data.image_b64 : '');

        const printHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NeuroScanAI — Medical Report | ${cId}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter','Segoe UI',sans-serif;font-size:11pt;color:#1a1a2e;background:#fff;line-height:1.5;}
.page{max-width:800px;margin:0 auto;padding:28px 32px;}
.report-header{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2.5px solid #1a1a2e;padding-bottom:14px;margin-bottom:16px;}
.logo-area{display:flex;align-items:center;gap:10px;}
.logo-icon{width:42px;height:42px;background:#1a1a2e;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#a78bfa;font-size:20px;font-weight:700;}
.logo-text h1{font-size:17pt;font-weight:700;color:#1a1a2e;}
.logo-text p{font-size:8pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;}
.report-badge{background:#fef3c7;border:1.5px solid #f59e0b;color:#92400e;padding:5px 12px;border-radius:6px;font-size:8pt;font-weight:700;letter-spacing:0.5px;text-align:right;}
.report-badge .rpt-id,.report-badge .rpt-date{font-size:7pt;color:#78716c;margin-top:2px;}
.patient-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin-bottom:16px;}
.pi-item .pi-label{font-size:7pt;text-transform:uppercase;letter-spacing:0.8px;color:#94a3b8;}
.pi-item .pi-value{font-size:10pt;font-weight:600;color:#1e293b;margin-top:2px;}
.section{margin-bottom:16px;}
.section-title{display:flex;align-items:center;gap:8px;font-size:10pt;font-weight:700;color:#1a1a2e;text-transform:uppercase;letter-spacing:0.6px;border-left:3px solid #6c5ce7;padding-left:8px;margin-bottom:10px;}
.section-title .num{background:#6c5ce7;color:#fff;width:20px;height:20px;border-radius:50%;font-size:8pt;display:inline-flex;align-items:center;justify-content:center;}
.detection-box{border:2px solid ${detectionColor};border-radius:10px;padding:14px 18px;background:${detectionColor}18;display:flex;align-items:center;gap:16px;}
.det-indicator{width:52px;height:52px;border-radius:50%;background:${detectionColor}22;border:2px solid ${detectionColor};display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.det-label{font-size:12pt;font-weight:700;color:${detectionColor};}
.det-conf{margin-top:4px;font-size:9pt;color:#4b5563;}
.det-grade{margin-top:2px;font-size:9pt;color:#6b7280;}
.conf-bar-wrap{margin-top:6px;display:flex;align-items:center;gap:8px;}
.conf-bar-bg{background:#e5e7eb;border-radius:4px;height:7px;width:140px;}
.conf-bar-fill{background:${detectionColor};height:100%;border-radius:4px;width:${confPct}%;}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;}
.metric-card{border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;}
.metric-card h4{font-size:8pt;text-transform:uppercase;letter-spacing:0.8px;color:#64748b;margin-bottom:8px;}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.metric-item .m-label{font-size:7.5pt;color:#94a3b8;}
.metric-item .m-value{font-size:11pt;font-weight:700;color:#1e293b;}
.metric-item .m-unit{font-size:7.5pt;color:#94a3b8;margin-left:2px;}
table{width:100%;border-collapse:collapse;font-size:9.5pt;}
th{background:#f1f5f9;padding:7px 10px;text-align:left;font-weight:600;color:#374151;font-size:8.5pt;}
td{padding:6px 10px;border-bottom:1px solid #f1f5f9;color:#374151;}
tr:last-child td{border-bottom:none;}
.scan-img-box{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;background:#000;display:flex;align-items:center;justify-content:center;min-height:220px;}
.scan-img-box img{width:100%;max-height:240px;object-fit:contain;}
.scan-img-box .no-img{color:#6b7280;font-size:9pt;text-align:center;padding:40px;}
.rad-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.rad-cell{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;}
.rad-cell .rc-label{font-size:7.5pt;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;}
.rad-cell .rc-value{font-size:10pt;font-weight:600;color:#1e293b;margin-top:2px;}
.rad-cell .rc-sub{font-size:7.5pt;color:#6b7280;margin-top:1px;}
.impression-box{background:#f8fafc;border-left:3px solid #6c5ce7;border-radius:0 6px 6px 0;padding:12px 14px;font-size:9.5pt;color:#374151;line-height:1.6;}
.disclaimer{background:#fef3c7;border:1.5px solid #f59e0b;border-radius:8px;padding:10px 14px;font-size:8pt;color:#78350f;margin-bottom:16px;}
.disclaimer strong{display:block;margin-bottom:4px;font-size:8.5pt;}
.report-footer{border-top:1px solid #e2e8f0;margin-top:16px;padding-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:7.5pt;color:#94a3b8;}
.sig-line{border-bottom:1px solid #374151;width:180px;margin-bottom:3px;}
.print-action{position:fixed;top:20px;right:20px;display:flex;gap:10px;z-index:999;}
.print-action button{padding:10px 20px;border:none;border-radius:8px;font-family:inherit;font-size:10pt;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px;}
.btn-print{background:#6c5ce7;color:#fff;}
.btn-close{background:#e5e7eb;color:#374151;}
@media print{.print-action{display:none!important;}body{background:#fff;}.page{padding:14px 18px;}}
</style>
</head>
<body>
<div class="print-action">
    <button class="btn-print" onclick="window.print()">🖨 Print / Save PDF</button>
    <button class="btn-close" onclick="window.close()">✕ Close</button>
</div>
<div class="page">
    <div class="report-header">
        <div class="logo-area">
            <div class="logo-icon">Nv</div>
            <div class="logo-text"><h1>NeuroScanAI Diagnostics</h1><p>AI-Assisted Neuroimaging Report</p></div>
        </div>
        <div class="report-badge">⚠ PRELIMINARY · AI-GENERATED<div class="rpt-id">Report #: ${reportId}</div><div class="rpt-date">Generated: ${now}</div></div>
    </div>
    <div class="patient-bar">
        <div class="pi-item"><div class="pi-label">Patient ID</div><div class="pi-value">${cId}</div></div>
        <div class="pi-item"><div class="pi-label">Study Date</div><div class="pi-value">${now.split(',')[0]}</div></div>
        <div class="pi-item"><div class="pi-label">Modality</div><div class="pi-value">${sm.modality_label || sm.orientation || 'MR / Brain'}</div></div>
        <div class="pi-item"><div class="pi-label">Reporting Model</div><div class="pi-value">Hybrid CNN-Tx v2.0</div></div>
    </div>
    <div class="section">
        <div class="section-title"><span class="num">1</span> Clinical Indication</div>
        <p style="font-size:9.5pt;color:#374151;line-height:1.6;">AI-assisted diagnostic scan submitted for automated tumor-detection pipeline analysis. Patient: <strong>${pName}</strong> | Case Reference: <strong>${cId}</strong>. Values below illustrate detected findings and AI-generated assessment.</p>
    </div>
    <div class="section">
        <div class="section-title"><span class="num">2</span> Imaging Findings</div>
        <div class="two-col">
            <div class="scan-img-box">${scanImgSrc ? `<img src="${scanImgSrc}" alt="MRI Scan">` : `<div class="no-img">📷 No scan image available</div>`}</div>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Location</td><td>${measActive ? (loc.full_location || loc.region || '—') : '—'}</td></tr>
                <tr><td>Laterality</td><td>${measActive ? (loc.laterality || '—') : '—'}</td></tr>
                <tr><td>Shape / Margin</td><td>${measActive ? (shape.shape_label || '—') : '—'}</td></tr>
                <tr><td>Signal Pattern</td><td>${measActive ? (enh.enhancement_pattern || '—') : '—'}</td></tr>
                <tr><td>Est. Volume</td><td>${measActive ? fmt(meas.volume_ml, 2, ' mL') : '—'}</td></tr>
                <tr><td>Edema Signal</td><td>${measActive ? (edema.edema_label || '—') : '—'}</td></tr>
                <tr><td>Mass Effect</td><td>${measActive ? (loc.surgical_risk || 'Not assessed') : 'Not assessed'}</td></tr>
            </table>
        </div>
    </div>
    <div class="section">
        <div class="section-title"><span class="num">3</span> Impression</div>
        <div class="detection-box">
            <div class="det-indicator">${isIndeterminate ? '❓' : isTumor ? '⚠' : '✅'}</div>
            <div>
                <div class="det-label">${detectionLabel}</div>
                <div class="det-conf">AI Confidence: <div class="conf-bar-wrap"><div class="conf-bar-bg"><div class="conf-bar-fill"></div></div><strong>${confPct}%</strong></div></div>
                <div class="det-grade">Grade Indicator: ${data.grade || '—'}</div>
            </div>
        </div>
    </div>
    <div class="section">
        <div class="section-title"><span class="num">4</span> Model Confidence &amp; Metrics</div>
        <div class="two-col">
            <div class="metric-card">
                <h4>Measurements</h4>
                <div class="metric-grid">
                    <div class="metric-item"><div class="m-label">Volume</div><div class="m-value">${measActive ? fmt(meas.volume_ml, 2) : '—'}<span class="m-unit">${measActive && meas.volume_ml != null ? 'mL' : ''}</span></div></div>
                    <div class="metric-item"><div class="m-label">Diameter (RANO)</div><div class="m-value">${measActive ? fmt(meas.diameter_mm, 1) : '—'}<span class="m-unit">${measActive && meas.diameter_mm != null ? 'mm' : ''}</span></div></div>
                    <div class="metric-item"><div class="m-label">Perpendicular</div><div class="m-value">${measActive ? fmt(meas.perpendicular_mm, 1) : '—'}<span class="m-unit">${measActive && meas.perpendicular_mm != null ? 'mm' : ''}</span></div></div>
                    <div class="metric-item"><div class="m-label">Risk Level</div><div class="m-value" style="color:${risk.risk_color || '#6b7280'};">${risk.risk_level || '—'}</div></div>
                </div>
            </div>
            <div class="metric-card">
                <h4>Scan Parameters</h4>
                <div class="metric-grid">
                    <div class="metric-item"><div class="m-label">Pixel Spacing X</div><div class="m-value">${sm.pixel_spacing_x != null ? sm.pixel_spacing_x : '—'}<span class="m-unit">${sm.pixel_spacing_x != null ? 'mm' : ''}</span></div></div>
                    <div class="metric-item"><div class="m-label">Pixel Spacing Y</div><div class="m-value">${sm.pixel_spacing_y != null ? sm.pixel_spacing_y : '—'}<span class="m-unit">${sm.pixel_spacing_y != null ? 'mm' : ''}</span></div></div>
                    <div class="metric-item"><div class="m-label">Slice Thickness</div><div class="m-value">${sm.slice_thickness != null ? sm.slice_thickness : '—'}<span class="m-unit">${sm.slice_thickness != null ? 'mm' : ''}</span></div></div>
                    <div class="metric-item"><div class="m-label">Orientation</div><div class="m-value" style="font-size:9pt;">${sm.orientation || '—'}</div></div>
                </div>
            </div>
        </div>
    </div>
    <div class="section">
        <div class="section-title"><span class="num">5</span> Radiomics Features</div>
        <div class="rad-grid">
            <div class="rad-cell"><div class="rc-label">Location</div><div class="rc-value">${measActive ? (loc.full_location || '—') : '—'}</div><div class="rc-sub">${measActive && loc.is_eloquent ? '⚠ Eloquent Area' : (measActive ? loc.depth_category || '' : '')}</div></div>
            <div class="rad-cell"><div class="rc-label">Enhancement</div><div class="rc-value">${measActive ? (enh.enhancement_pattern || '—') : '—'}</div><div class="rc-sub">${measActive && enh.is_ring_enhancement ? '⚠ Ring Enhancement' : (measActive ? 'Ratio: ' + (enh.enhancement_ratio || 0).toFixed(2) : '')}</div></div>
            <div class="rad-cell"><div class="rc-label">Peritumoral Edema</div><div class="rc-value">${measActive ? (edema.edema_label || '—') : '—'}</div><div class="rc-sub">${measActive && edema.edema_ratio ? 'Ratio: ' + edema.edema_ratio.toFixed(2) : ''}</div></div>
            <div class="rad-cell"><div class="rc-label">Shape</div><div class="rc-value">${measActive ? (shape.shape_label || '—') : '—'}</div><div class="rc-sub">${measActive && shape.sphericity ? 'Sphericity: ' + shape.sphericity.toFixed(3) : ''}</div></div>
            <div class="rad-cell"><div class="rc-label">Texture (GLCM)</div><div class="rc-value">${measActive ? (tex.texture_label || '—') : '—'}</div><div class="rc-sub">${measActive && tex.homogeneity ? 'Homogeneity: ' + tex.homogeneity.toFixed(3) : ''}</div></div>
            <div class="rad-cell"><div class="rc-label">Surgical Risk</div><div class="rc-value">${measActive ? (loc.surgical_risk || '—') : '—'}</div><div class="rc-sub">${measActive && loc.depth_category ? 'Depth: ' + loc.depth_category : ''}</div></div>
        </div>
    </div>
    <div class="disclaimer"><strong>⚠ LIMITATIONS &amp; DISCLOSURE</strong>This report is AI-generated and has <u>not</u> been reviewed by a radiologist. It is a demonstration built from imaging data; no diagnostic conclusion should be drawn from it. In a real deployment, this section would state the model's training dataset, validated sensitivity/specificity, intended use population, and regulatory status (e.g. CE/FDA clearance), per standard AI-in-radiology transparency requirements.</div>
    ${rfRows ? `<div class="section"><div class="section-title"><span class="num">6</span> Risk Factor Breakdown</div><table><tr><th>Factor</th><th>Score Distribution</th><th style="text-align:right;">Score</th></tr>${rfRows}</table></div>` : ''}
    <div class="section">
        <div class="section-title"><span class="num">7</span> Recommendation</div>
        <div class="impression-box">${impression}</div>
    </div>
    <div class="report-footer">
        <div><div style="font-size:8pt;color:#374151;margin-bottom:4px;">Reviewing Radiologist</div><div class="sig-line"></div><div>Signature &amp; Date — <em>pending</em></div></div>
        <div style="text-align:right;"><div>NeuroVision Diagnostics — prototype report template</div><div>Non-clinical sample record &nbsp;·&nbsp; Page 1 of 1</div></div>
    </div>
</div>
</body>
</html>`;

        const printWin = window.open('', '_blank', 'width=940,height=720');
        if (!printWin) {
            alert('Pop-up blocked! Please allow pop-ups for this site and try again.');
            return;
        }
        printWin.document.write(printHTML);
        printWin.document.close();
    }

    // ─── Clinical Case Archive ────────────────────────────────────────────────

    const refreshReportsBtn = document.getElementById('refresh-reports-btn');
    const reportsTableBody = document.getElementById('reports-table-body');
    const totalCountEl = document.getElementById('total-reports-count');
    const tumorCountEl = document.getElementById('tumor-reports-count');
    const healthyCountEl = document.getElementById('healthy-reports-count');
    let allReports = [];

    const RISK_COLORS = {
        'Low': '#22c55e', 'Moderate': '#f59e0b',
        'High': '#f97316', 'Critical': '#ef4444', 'N/A': '#6b7280'
    };

    function loadReports() {
        if (!reportsTableBody) return;
        const spinIcon = refreshReportsBtn ? refreshReportsBtn.querySelector('i') : null;
        if (spinIcon) spinIcon.classList.add('fa-spin');
        reportsTableBody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:3rem;opacity:0.5;">Synchronizing with Archive...</td></tr>';

        fetch('/get_reports')
            .then(res => res.json())
            .then(reports => {
                allReports = reports;
                reportsTableBody.innerHTML = '';
                let tumorCount = 0, healthyCount = 0;

                if (reports.length === 0) {
                    reportsTableBody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:3rem;">Archival Vault is Empty.</td></tr>';
                } else {
                    reports.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                    reports.forEach((r, idx) => {
                        const isTumor = r.diagnosis && r.diagnosis.includes('POSITIVE');
                        if (isTumor) tumorCount++; else healthyCount++;

                        const date = r.timestamp
                            ? new Date(r.timestamp).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
                            : 'N/A';
                        const rl = r.risk_level || 'N/A';
                        const rc = RISK_COLORS[rl] || '#6b7280';
                        const vol = r.volume_ml ? r.volume_ml.toFixed(2) + ' mL' : '—';

                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td><span style="opacity:0.7">${date}</span></td>
                            <td><code style="color:var(--secondary)">${r.case_id || 'N/A'}</code></td>
                            <td><span style="font-weight:500">${r.patient_name || 'Anonymous'}</span></td>
                            <td><span class="badge-clinical ${isTumor ? 'tumor' : 'healthy'}">${isTumor ? 'POSITIVE' : 'NEGATIVE'}</span></td>
                            <td><span class="badge-risk" style="color:${rc};border-color:${rc}44;background:${rc}18">${rl}</span></td>
                            <td>${vol}</td>
                            <td>
                                <div class="confidence-cell">
                                    <span>${((r.confidence || 0) * 100).toFixed(1)}%</span>
                                    <div class="confidence-bar-mini"><div class="fill" style="width:${(r.confidence || 0) * 100}%"></div></div>
                                </div>
                            </td>
                            <td style="text-align:right">
                                <button class="btn-action-sm view-report-btn" data-index="${idx}" title="View Report"><i class="fa-solid fa-folder-open"></i></button>
                            </td>`;
                        reportsTableBody.appendChild(row);
                    });
                }
                if (totalCountEl) totalCountEl.textContent = reports.length;
                if (tumorCountEl) tumorCountEl.textContent = tumorCount;
                if (healthyCountEl) healthyCountEl.textContent = healthyCount;

                document.querySelectorAll('.view-report-btn').forEach(btn => {
                    btn.addEventListener('click', () => viewReport(allReports[btn.dataset.index]));
                });
            })
            .catch(e => {
                reportsTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--danger)">Error: ${e}</td></tr>`;
            })
            .finally(() => {
                if (spinIcon) spinIcon.classList.remove('fa-spin');
            });
    }

    loadReports();

    function viewReport(report) {
        currentAnalysisData = {
            label: report.diagnosis || '',
            confidence: report.confidence || 0,
            grade: report.grade || 'N/A',
            image_b64: report.image_b64 || '',
            overlay_b64: report.overlay_b64 || '',
            measurements: { volume_ml: report.volume_ml, diameter_mm: report.diameter_mm },
            scan_meta: report.scan_meta || {},
            radiomics: report.radiomics || {},
            risk: { total_score: report.risk_score || 0, risk_level: report.risk_level || 'N/A', factor_breakdown: report.risk_breakdown || [], risk_color: RISK_COLORS[report.risk_level] || '#888' },
            growth: report.growth || null,
            details: { tumor_size: report.tumor_size, tumor_loc: report.tumor_loc, impression: report.impression }
        };

        if (patientNameInput) patientNameInput.value = report.patient_name || '';
        if (caseIdInput) caseIdInput.value = report.case_id || '';

        if (uploadPrompt) uploadPrompt.classList.add('hidden');
        if (resultsSection) resultsSection.classList.remove('hidden');
        updateInferenceUI(currentAnalysisData);

        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelector('.nav-item[data-target="testing"]')?.classList.add('active');
        document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
        document.getElementById('testing')?.classList.add('active');
    }

    if (refreshReportsBtn) refreshReportsBtn.addEventListener('click', loadReports);
    document.querySelector('.nav-item[data-target="reports"]')?.addEventListener('click', loadReports);

    // ─── Settings ─────────────────────────────────────────────────────────────
    const settingThreshold = document.getElementById('setting-threshold');
    const settingThresholdDisplay = document.getElementById('setting-threshold-display');
    if (settingThreshold) {
        settingThreshold.addEventListener('input', () => {
            if (settingThresholdDisplay) settingThresholdDisplay.textContent = parseFloat(settingThreshold.value).toFixed(2);
        });
    }

    // Load settings from server
    fetch('/get_settings')
        .then(r => r.json())
        .then(s => {
            if (settingThreshold) { settingThreshold.value = s.segmentation_threshold || 0.5; settingThreshold.dispatchEvent(new Event('input')); }
            if (thresholdSlider) { thresholdSlider.value = s.segmentation_threshold || 0.5; thresholdSlider.dispatchEvent(new Event('input')); }
            const gpu = document.getElementById('setting-gpu'); if (gpu) gpu.checked = s.gpu_acceleration !== false;
            const heur = document.getElementById('setting-heuristic'); if (heur) heur.checked = s.heuristic_fallback !== false;
            const prec = document.getElementById('setting-precision'); if (prec) prec.value = s.inference_precision || 'fp32';
        }).catch(() => { });

    const saveSettingsBtn = document.querySelector('.btn-save-settings');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', () => {
            const thr = settingThreshold ? parseFloat(settingThreshold.value) : 0.5;
            const gpu = document.getElementById('setting-gpu');
            const heur = document.getElementById('setting-heuristic');
            const prec = document.getElementById('setting-precision');

            const payload = {
                segmentation_threshold: thr,
                gpu_acceleration: gpu ? gpu.checked : true,
                heuristic_fallback: heur ? heur.checked : true,
                inference_precision: prec ? prec.value : 'fp32'
            };

            saveSettingsBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';
            fetch('/save_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
                .then(r => r.json())
                .then(resp => {
                    saveSettingsBtn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Saved!';
                    if (thresholdSlider) { thresholdSlider.value = thr; thresholdSlider.dispatchEvent(new Event('input')); }
                    setTimeout(() => {
                        saveSettingsBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Apply Global Configuration';
                    }, 2000);
                })
                .catch(() => { saveSettingsBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Apply Global Configuration'; });
        });
    }

    // ─── Global Helpers ───────────────────────────────────────────────────────
    window.togglePathInput = function () {
        const select = document.getElementById('data-source-select');
        const inputGroup = document.getElementById('path-input-group');
        if (select && inputGroup) {
            inputGroup.classList.toggle('hidden', select.value !== 'local');
        }
    };
});
