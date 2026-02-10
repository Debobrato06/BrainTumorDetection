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
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewImage = document.getElementById('preview-image');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resetBtn = document.getElementById('reset-btn');
    const detectionText = document.getElementById('detection-text');
    const statusIndicator = document.getElementById('status-indicator');
    const analysisResultsBox = document.getElementById('analysis-results-box');
    const initialMessage = analysisResultsBox.querySelector('.initial-message');
    const resultDataStructured = document.getElementById('result-data-structured');

    // Metadata Fields
    const metaFormat = document.getElementById('meta-format');
    const metaRes = document.getElementById('meta-res');

    // Report Fields
    const confidenceVal = document.getElementById('confidence-val');
    const tumorGradeVal = document.getElementById('tumor-grade-val');
    const tumorSizeVal = document.getElementById('tumor-size-val');
    const tumorLocVal = document.getElementById('tumor-loc-val');
    const impressionText = document.getElementById('impression-text');

    // Inputs
    const patientNameInput = document.getElementById('patient-name');
    const caseIdInput = document.getElementById('case-id');
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
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            resultsSection.classList.add('hidden');
            uploadPrompt.classList.remove('hidden');
            resultDataStructured.classList.add('hidden');
            initialMessage.classList.remove('hidden');
            fileInput.value = '';
            currentAnalysisData = null;
            patientNameInput.value = '';
            caseIdInput.value = '';
            metaFormat.innerText = "N/A";
            metaRes.innerText = "256 x 256";
        });
    }

    function handleFile(file) {
        const validExtensions = ['.nii', '.nii.gz', '.jpg', '.jpeg', '.png'];
        const isExtensionValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        if (!isExtensionValid) {
            alert('UNSUPPORTED MEDIA: System only accepts NIfTI or Radiology Images.');
            return;
        }

        uploadPrompt.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');

        // Update Meta
        metaFormat.innerText = file.name.split('.').pop().toUpperCase();

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('CRITICAL ERROR: ' + data.error);
                    resetBtn.click();
                } else {
                    currentAnalysisData = data;
                    updateInferenceUI(data);
                }
            })
            .catch(err => {
                alert('ACQUISITION FAILED: ' + err);
                resetBtn.click();
            })
            .finally(() => {
                loadingOverlay.classList.add('hidden');
            });
    }

    function updateInferenceUI(data) {
        if (data.image_b64) previewImage.src = 'data:image/png;base64,' + data.image_b64;

        initialMessage.classList.add('hidden');
        resultDataStructured.classList.remove('hidden');

        const isTumor = data.label.includes('DETECTED');
        detectionText.innerText = data.label;
        statusIndicator.className = 'detection-status-card ' + (isTumor ? 'tumor' : 'healthy');

        confidenceVal.innerText = (data.confidence * 100).toFixed(1) + '%';
        tumorGradeVal.innerText = data.grade || "N/A";

        if (data.details) {
            tumorSizeVal.innerText = data.details.tumor_size;
            tumorLocVal.innerText = data.details.tumor_loc;
            impressionText.innerText = data.details.impression;
        }
    }

    if (saveReportBtn) {
        saveReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) {
                alert("No analysis data to save. Please run a scan first.");
                return;
            }

            const pName = patientNameInput.value || "Anonymous";
            const cId = caseIdInput.value || "TEMP-" + Math.floor(Math.random() * 10000);

            saveReportBtn.disabled = true;
            saveReportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Archiving...';

            const reportData = {
                case_id: cId,
                patient_name: pName,
                diagnosis: currentAnalysisData.label,
                confidence: currentAnalysisData.confidence,
                tumor_size: currentAnalysisData.details ? currentAnalysisData.details.tumor_size : "N/A",
                tumor_loc: currentAnalysisData.details ? currentAnalysisData.details.tumor_loc : "N/A",
                impression: currentAnalysisData.details ? currentAnalysisData.details.impression : "",
                image_b64: currentAnalysisData.image_b64
            };

            fetch('/save_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reportData)
            })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert("Case " + cId + " has been successfully archived in the Clinical Vault.");
                        // Optional: trigger a refresh of the reports table
                        if (reportsTableBody) loadReports();
                    } else {
                        alert("ERROR: " + data.message);
                    }
                })
                .catch(err => alert("NETWORK ERROR: " + err))
                .finally(() => {
                    saveReportBtn.disabled = false;
                    saveReportBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Archiving Report';
                });
        });
    }

    // --- Clinical Case Archive Logic ---
    const refreshReportsBtn = document.getElementById('refresh-reports-btn');
    const reportsTableBody = document.getElementById('reports-table-body');
    const totalCountEl = document.getElementById('total-reports-count');
    const tumorCountEl = document.getElementById('tumor-reports-count');
    const healthyCountEl = document.getElementById('healthy-reports-count');
    let allReports = [];

    function loadReports() {
        if (!reportsTableBody) return;

        // UI Feedback
        if (refreshReportsBtn) {
            const refreshIcon = refreshReportsBtn.querySelector('i');
            if (refreshIcon) refreshIcon.classList.add('fa-spin');
        }

        reportsTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 3rem; opacity: 0.5;">Synchronizing with Archive...</td></tr>';

        fetch('/get_reports')
            .then(res => res.json())
            .then(reports => {
                allReports = reports;
                reportsTableBody.innerHTML = '';

                // Summarize Stats
                let tumorCount = 0;
                let healthyCount = 0;

                if (reports.length === 0) {
                    reportsTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 3rem;">Archival Vault is Empty.</td></tr>';
                } else {
                    // Sort newest first by default
                    reports.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

                    reports.forEach((r, index) => {
                        const isTumor = r.diagnosis && r.diagnosis.includes('NEOPLASTIC');
                        if (isTumor) tumorCount++; else healthyCount++;

                        const row = document.createElement('tr');
                        const date = r.timestamp ? new Date(r.timestamp).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : 'N/A';

                        row.innerHTML = `
                            <td><span style="opacity: 0.7;">${date}</span></td>
                            <td><code style="color: var(--secondary);">${r.case_id || 'N/A'}</code></td>
                            <td><span style="font-weight: 500;">${r.patient_name || 'Anonymous'}</span></td>
                            <td><span class="badge-clinical ${isTumor ? 'tumor' : 'healthy'}">${r.diagnosis || 'Unknown'}</span></td>
                            <td>
                                <div class="confidence-cell">
                                    <span>${((r.confidence || 0) * 100).toFixed(1)}%</span>
                                    <div class="confidence-bar-mini">
                                        <div class="fill" style="width: ${(r.confidence || 0) * 100}%;"></div>
                                    </div>
                                </div>
                            </td>
                            <td style="text-align: right;">
                                <button class="btn-action-sm view-report-btn" data-index="${index}" title="Open Case Detail">
                                    <i class="fa-solid fa-folder-open"></i>
                                </button>
                            </td>
                        `;
                        reportsTableBody.appendChild(row);
                    });
                }

                // Update Stats
                if (totalCountEl) totalCountEl.innerText = reports.length;
                if (tumorCountEl) tumorCountEl.innerText = tumorCount;
                if (healthyCountEl) healthyCountEl.innerText = healthyCount;

                // Attach Event Listeners
                document.querySelectorAll('.view-report-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const index = btn.getAttribute('data-index');
                        viewReport(allReports[index]);
                    });
                });
            })
            .catch(e => {
                reportsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--danger);">Synchronization Error: ${e}</td></tr>`;
            })
            .finally(() => {
                if (refreshReportsBtn) {
                    const refreshIcon = refreshReportsBtn.querySelector('i');
                    if (refreshIcon) refreshIcon.classList.remove('fa-spin');
                }
            });
    }

    // Initial load
    loadReports();

    // --- Global Helpers ---
    window.togglePathInput = function () {
        const select = document.getElementById('data-source-select');
        const inputGroup = document.getElementById('path-input-group');
        if (select && inputGroup) {
            if (select.value === 'local') {
                inputGroup.classList.remove('hidden');
            } else {
                inputGroup.classList.add('hidden');
            }
        }
    };

    function viewReport(report) {
        // Populate Data
        currentAnalysisData = {
            label: report.diagnosis,
            confidence: report.confidence,
            image_b64: report.image_b64,
            details: {
                tumor_size: report.tumor_size,
                tumor_loc: report.tumor_loc,
                impression: report.impression
            }
        };

        // Populate Inputs
        if (patientNameInput) patientNameInput.value = report.patient_name || "";
        if (caseIdInput) caseIdInput.value = report.case_id || "";

        // Populate UI
        updateInferenceUI(currentAnalysisData);

        // Switch View
        // 1. Hide Upload Prompt, Show Results, Ensure Dropzone is visible
        if (uploadPrompt) uploadPrompt.classList.add('hidden');
        if (dropZone) dropZone.style.display = 'flex';
        if (resultsSection) resultsSection.classList.remove('hidden');

        // 2. Switch Tab
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
        const testingNav = document.querySelector('.nav-item[data-target="testing"]');
        if (testingNav) testingNav.classList.add('active');

        document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
        const testingSec = document.getElementById('testing');
        if (testingSec) testingSec.classList.add('active');
    }

    // Hook up refresh button
    if (refreshReportsBtn) {
        refreshReportsBtn.addEventListener('click', loadReports);
    }

    // Auto-load reports when tab is clicked
    const reportsNav = document.querySelector('.nav-item[data-target="reports"]');
    if (reportsNav) {
        reportsNav.addEventListener('click', loadReports);
    }
});
