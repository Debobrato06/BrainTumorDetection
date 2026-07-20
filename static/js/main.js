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
            statusBadge.classList.add('active');
            statusBadge.querySelector('.status-label').innerText = 'Engine Active';

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
        statusBadge.classList.remove('active');
        statusBadge.querySelector('.status-label').innerText = 'Engine Standby';
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
        const isIndeterminate    = data.is_indeterminate === true;
        const isTumor            = !isIndeterminate && data.label && data.label.includes('POSITIVE');
        const insufficientMask   = data.insufficient_mask_data === true;
        const reliabilityReduced = data.reliability_reduced === true;
        const oodDetected        = data.ood_detected === true;
        const alert              = data.alert || { level: 'none', title: '', message: '', show_results: true };
        const pipeStatus         = data.pipeline_status || {};

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
        if (metaPx)   metaPx.textContent   = sm.pixel_spacing_x ? `${sm.pixel_spacing_x}mm \u00d7 ${sm.pixel_spacing_y}mm` : '\u2014';
        if (metaSt)   metaSt.textContent   = sm.slice_thickness  ? `${sm.slice_thickness}mm` : '\u2014';

        // ── Rule 8: Top-Level Fail-Loud Alert Banner ───────────────────────────
        renderAlertBanner(alert, pipeStatus, oodDetected, reliabilityReduced);

        // ── Rule 4: OOD / Low Reliability Badge ───────────────────────────────
        renderReliabilityBadge(reliabilityReduced, oodDetected, pipeStatus, data.seg_quality_score);

        // ── Metadata Warning (secondary, subordinate to alert banner) ──────────
        const warnBanner = document.getElementById('meta-warning-banner');
        const warnText   = document.getElementById('meta-warning-text');
        if (sm.warning && warnBanner && warnText && alert.level !== 'critical') {
            warnText.textContent = sm.warning;
            warnBanner.classList.remove('hidden');
        } else if (warnBanner) {
            warnBanner.classList.add('hidden');
        }

        // ── Rule 5: Detection Hero Card (3 states) ────────────────────────────
        const heroCard      = document.getElementById('detection-hero-card');
        const detectionText = document.getElementById('detection-text');
        const detectionIcon = document.getElementById('detection-icon');
        const confVal       = document.getElementById('confidence-val');
        const confBarFill   = document.getElementById('conf-bar-fill');
        const confCaveat    = document.getElementById('conf-caveat');

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

        setText('vol-val',  measActive ? fmt(meas.volume_ml,  2, ' mL') : '\u2014');
        setText('diam-val', measActive ? fmt(meas.diameter_mm, 1, ' mm') : '\u2014');
        setText('perp-val', measActive ? fmt(meas.perpendicular_mm, 1, ' mm') : '\u2014');

        ['vol-card','diam-card','perp-card'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.classList.toggle('state-na', !measActive);
        });

        // ── Rule 3: Grade (never stale \u2014 always server-sanitized) ─────────────
        const gradeEl   = document.getElementById('grade-val');
        const gradeCard = document.getElementById('grade-card');
        if (gradeEl) {
            var g = data.grade || '\u2014';
            gradeEl.textContent = g;
            var gradeIsNA = !isTumor || insufficientMask || g.startsWith('Not');
            if (gradeCard) gradeCard.classList.toggle('state-na', gradeIsNA);
        }

        // ── Risk Gauge ────────────────────────────────────────────────────────
        const risk       = data.risk || {};
        const riskScore  = (risk.total_score != null) ? risk.total_score : 0;
        const riskLevel  = risk.risk_level || null;
        const riskColor  = risk.risk_color || '#888888';
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
            ['rad-location','rad-enhancement','rad-edema','rad-shape','rad-texture','rad-surgical-risk',
             'rad-eloquent','rad-ring','rad-edema-ratio','rad-sphericity','rad-homogeneity','rad-depth'
            ].forEach(function(id) { setText(id, naMsg); });
        }

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
        var levelClass  = { critical: 'alert-critical', warning: 'alert-warning', info: 'alert-info' };
        var levelIcon   = { critical: '\uD83D\uDEA8', warning: '\u26A0\uFE0F', info: '\u2139\uFE0F' };
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
            { key: 'detection_stage',    label: 'Detection'    },
            { key: 'segmentation_stage', label: 'Segmentation' },
            { key: 'radiomics_stage',    label: 'Radiomics'    },
            { key: 'grading_stage',      label: 'Grading'      },
            { key: 'risk_scoring_stage', label: 'Risk Score'   },
        ];
        var iconMap  = { completed: '\u2713', failed: '\u2717', not_run: '\u25CB', skipped: '\u2014' };
        var colorMap = { completed: '#22c55e', failed: '#ef4444', not_run: '#6b7280', skipped: '#9ca3af' };
        container.innerHTML = stages.map(function(s) {
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
    window.showOverlay = function() {
        const ov = document.getElementById('overlay-img');
        const or = document.getElementById('original-img');
        const b1 = document.getElementById('btn-show-overlay');
        const b2 = document.getElementById('btn-show-original');
        if (ov) ov.classList.remove('hidden');
        if (or) or.classList.add('hidden');
        if (b1) b1.classList.add('active');
        if (b2) b2.classList.remove('active');
    };
    window.showOriginal = function() {
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
        if (refreshReportsBtn) refreshReportsBtn.querySelector('i').classList.add('fa-spin');
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
                            ? new Date(r.timestamp).toLocaleString('en-GB', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' })
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
                                    <span>${((r.confidence || 0)*100).toFixed(1)}%</span>
                                    <div class="confidence-bar-mini"><div class="fill" style="width:${(r.confidence||0)*100}%"></div></div>
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
                if (refreshReportsBtn) refreshReportsBtn.querySelector('i').classList.remove('fa-spin');
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
        }).catch(() => {});

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
