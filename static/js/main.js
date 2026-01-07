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

    // --- Dashboard Chart ---
    const ctx = document.getElementById('accuracyChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6'],
                datasets: [{
                    label: 'Accuracy',
                    data: [65, 72, 78, 81, 85, 88],
                    borderColor: '#6c5ce7',
                    backgroundColor: 'rgba(108, 92, 231, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#a0a3b1' } }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: '#2d3446' },
                        ticks: { color: '#a0a3b1' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#a0a3b1' }
                    }
                }
            }
        });
    }

    // --- Training Logic ---
    const trainingForm = document.getElementById('training-form');
    const startBtn = document.getElementById('start-training-btn');
    const logsContainer = document.getElementById('training-logs');
    const progressBar = document.getElementById('train-progress');
    let pollInterval = null;

    if (trainingForm) {
        trainingForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(trainingForm);
            const data = Object.fromEntries(formData.entries());

            // UI Updates
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Initializing...';
            logsContainer.innerHTML = '<div class="log-line system">> Initializing Training Sequence...</div>';

            fetch('/start_training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
                .then(res => res.json())
                .then(res => {
                    if (res.status === 'success') {
                        startBtn.innerText = 'Training in Progress...';
                        startPolling();
                    } else {
                        alert(res.message);
                        startBtn.disabled = false;
                        startBtn.innerText = 'Start Training';
                    }
                });
        });
    }

    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(() => {
            fetch('/training_status')
                .then(res => res.json())
                .then(data => {
                    // Update Progress
                    progressBar.style.width = data.progress + '%';

                    // Update Logs
                    if (data.logs.length > 0) {
                        logsContainer.innerHTML = '';
                        data.logs.forEach(log => {
                            const div = document.createElement('div');
                            div.className = 'log-line';
                            div.innerText = '> ' + log;
                            logsContainer.appendChild(div);
                        });
                        logsContainer.scrollTop = logsContainer.scrollHeight;
                    }

                    // Check if done
                    if (!data.is_running && data.progress >= 100) {
                        clearInterval(pollInterval);
                        startBtn.disabled = false;
                        startBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Training';
                        logsContainer.innerHTML += '<div class="log-line system" style="color:#55efc4">> PROCESS COMPLETE</div>';
                    }
                });
        }, 1000);
    }

    // --- Improved Testing / Inference Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    const previewImage = document.getElementById('preview-image');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resetBtn = document.getElementById('reset-btn');
    const detectionText = document.getElementById('detection-text');
    const statusIndicator = document.getElementById('status-indicator');

    // New Fields
    const confidenceVal = document.getElementById('confidence-val');
    const tumorSizeVal = document.getElementById('tumor-size-val');
    const tumorLocVal = document.getElementById('tumor-loc-val');
    const impressionText = document.getElementById('impression-text');

    // Inputs for Saving
    const patientNameInput = document.getElementById('patient-name');
    const caseIdInput = document.getElementById('case-id');
    const saveReportBtn = document.getElementById('save-report-btn');

    let currentAnalysisData = null; // Store current result to save later

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
            dropZone.style.display = 'flex';
            fileInput.value = '';
            currentAnalysisData = null;
            patientNameInput.value = '';
            caseIdInput.value = '';
        });
    }

    // Save Button Logic
    if (saveReportBtn) {
        saveReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) return alert("No analysis to save.");
            const name = patientNameInput.value.trim() || "Anonymous";
            const caseId = caseIdInput.value.trim() || ("CASE-" + Date.now().toString().slice(-6));

            const reportPayload = {
                case_id: caseId,
                patient_name: name,
                diagnosis: currentAnalysisData.label,
                confidence: currentAnalysisData.confidence,
                tumor_size: currentAnalysisData.details ? currentAnalysisData.details.tumor_size : 'N/A',
                tumor_loc: currentAnalysisData.details ? currentAnalysisData.details.tumor_loc : 'N/A',
                impression: currentAnalysisData.details ? currentAnalysisData.details.impression : 'N/A',
                image_b64: currentAnalysisData.image_b64
            };

            saveReportBtn.disabled = true;
            saveReportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

            fetch('/save_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reportPayload)
            })
                .then(res => res.json())
                .then(res => {
                    if (res.status === 'success') {
                        alert("Report saved successfully!");
                        loadReports();
                    } else {
                        alert("Error saving: " + res.message);
                    }
                })
                .catch(e => alert("Network Error: " + e))
                .finally(() => {
                    saveReportBtn.disabled = false;
                    saveReportBtn.innerHTML = '<i class="fa-solid fa-save"></i> Save to History';
                });
        });
    }

    function handleFile(file) {
        const validExtensions = ['.nii', '.nii.gz', '.jpg', '.jpeg', '.png'];
        const isExtensionValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        if (!isExtensionValid) {
            alert('Please upload a valid file (.nii, .nii.gz, .jpg, .png)');
            return;
        }

        dropZone.style.display = 'none';
        resultsSection.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Analysis Error: ' + data.error);
                    loadingOverlay.classList.add('hidden');
                    dropZone.style.display = 'flex';
                    resultsSection.classList.add('hidden');
                } else {
                    currentAnalysisData = data;
                    updateInferenceUI(data);
                }
            })
            .catch(err => {
                alert('Request Failed: ' + err);
                loadingOverlay.classList.add('hidden');
                dropZone.style.display = 'flex';
                resultsSection.classList.add('hidden');
            })
            .finally(() => {
                loadingOverlay.classList.add('hidden');
            });
    }

    function updateInferenceUI(data) {
        if (data.image_b64) previewImage.src = 'data:image/png;base64,' + data.image_b64;

        const isTumor = data.label === 'Tumor Detected';
        detectionText.innerText = data.label;
        statusIndicator.className = 'status-indicator ' + (isTumor ? 'tumor' : 'healthy');

        confidenceVal.innerText = (data.confidence * 100).toFixed(1) + '%';

        // Populate new details
        if (data.details) {
            tumorSizeVal.innerText = data.details.tumor_size;
            tumorLocVal.innerText = data.details.tumor_loc;
            impressionText.innerText = data.details.impression;
        }
    }

    // --- Reports / History Logic ---
    const refreshReportsBtn = document.getElementById('refresh-reports-btn');
    const reportsTableBody = document.getElementById('reports-table-body');
    let allReports = [];

    function loadReports() {
        if (!reportsTableBody) return;
        reportsTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">Loading...</td></tr>';

        fetch('/get_reports')
            .then(res => res.json())
            .then(reports => {
                allReports = reports;
                reportsTableBody.innerHTML = '';
                if (reports.length === 0) {
                    reportsTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No records found.</td></tr>';
                    return;
                }
                reports.forEach((r, index) => {
                    const row = document.createElement('tr');
                    const date = new Date(r.timestamp).toLocaleDateString() + ' ' + new Date(r.timestamp).toLocaleTimeString();
                    row.innerHTML = `
                        <td>${date}</td>
                        <td><strong>${r.case_id}</strong></td>
                        <td>${r.patient_name}</td>
                        <td><span class="badge ${r.diagnosis === 'Tumor Detected' ? 'badge-danger' : 'badge-success'}">${r.diagnosis}</span></td>
                        <td>${(r.confidence * 100).toFixed(1)}%</td>
                        <td>
                            <button class="btn-sm view-report-btn" data-index="${index}"><i class="fa-solid fa-eye"></i></button>
                        </td>
                    `;
                    reportsTableBody.appendChild(row);
                });

                // Attach Event Listeners
                document.querySelectorAll('.view-report-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const index = btn.getAttribute('data-index');
                        viewReport(allReports[index]);
                    });
                });
            })
            .catch(e => {
                reportsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:red;">Error loading reports: ${e}</td></tr>`;
            });
    }

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
        // 1. Hide Dropzone, Show Results
        if (dropZone) dropZone.style.display = 'none';
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
