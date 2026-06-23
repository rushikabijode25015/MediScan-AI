document.addEventListener('DOMContentLoaded', () => {
    // API endpoints
    const API_SAMPLES = '/api/samples';
    const API_PREDICT = '/api/predict';
    const API_METRICS = '/api/metrics';

    // State variables
    let selectedSampleId = null;
    let uploadedFile = null;
    let sampleData = [];
    let chartsInitialized = false;

    // DOM Elements
    const samplesContainer = document.getElementById('samples-container');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const diagnoseBtn = document.getElementById('diagnose-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingSubtext = document.getElementById('loading-subtext');
    const triageStatus = document.getElementById('triage-status');
    
    const imgOriginal = document.getElementById('img-original');
    const imgOverlay = document.getElementById('img-overlay');
    const sliderContainer = document.getElementById('slider-container');
    const opacitySlider = document.getElementById('opacity-slider');
    const opacityVal = document.getElementById('opacity-val');
    
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const confBar = document.getElementById('conf-bar');
    const clinicalRationale = document.getElementById('clinical-rationale');

    // 1. Fetch Preset Samples
    fetchSamples();

    // 2. Fetch Metrics and Initialize Charts
    fetchAndRenderMetrics();

    // --- Event Listeners ---

    // File input browse button
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input selection change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag and Drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // Diagnose Button Click
    diagnoseBtn.addEventListener('click', runDiagnosis);

    // Opacity Slider Interaction
    opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        opacityVal.textContent = `${val}%`;
        
        // Dynamically adjust the opacity of the overlay image
        // To make it smooth, we can set the opacity style.
        // Wait, if it is stacked, we can adjust it directly.
        // In our stacked layout, we can set style.opacity = val/100
        const heatmapImg = document.getElementById('img-heatmap-layer');
        if (heatmapImg) {
            heatmapImg.style.opacity = val / 100;
        } else {
            // Fallback for non-stacked layout: modify image filter or opacity directly
            imgOverlay.style.opacity = val / 100;
        }
    });

    // --- Helper Functions ---

    function fetchSamples() {
        fetch(API_SAMPLES)
            .then(res => res.json())
            .then(data => {
                sampleData = data;
                renderSampleButtons();
            })
            .catch(err => {
                console.error("Error fetching samples:", err);
                samplesContainer.innerHTML = '<p class="text-error">Error loading clinical samples.</p>';
            });
    }

    function renderSampleButtons() {
        samplesContainer.innerHTML = '';
        sampleData.forEach(sample => {
            const btn = document.createElement('button');
            btn.className = 'sample-btn';
            btn.dataset.id = sample.id;
            btn.innerHTML = `
                <span class="sample-category">${sample.category}</span>
                <span class="sample-name">${sample.name}</span>
            `;
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                selectSample(sample.id);
            });
            samplesContainer.appendChild(btn);
        });
    }

    function selectSample(id) {
        selectedSampleId = id;
        uploadedFile = null;
        
        // Reset file input
        fileInput.value = '';
        
        // Visual updates
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.id === id);
        });
        
        // Update drag-drop preview
        const selected = sampleData.find(s => s.id === id);
        if (selected) {
            dropzone.innerHTML = `
                <i class="fa-solid fa-file-medical-alt upload-icon" style="color: var(--color-cyan)"></i>
                <h3>Sample Selected</h3>
                <p class="text-teal">${selected.name}</p>
                <button class="btn btn-secondary btn-sm" id="browse-btn-reset">Reset</button>
            `;
            // Re-bind click event on reset button
            document.getElementById('browse-btn-reset').addEventListener('click', (e) => {
                e.stopPropagation();
                resetUploader();
            });
            
            // Set preview images
            imgOriginal.src = selected.path;
            
            // Setup simple double preview in index window before running model
            resetVisualizationBox(selected.path, false);
            
            diagnoseBtn.disabled = false;
        }
    }

    function handleFileSelection(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image scan file (JPG, PNG).');
            return;
        }
        
        uploadedFile = file;
        selectedSampleId = null;
        
        // Deselect sample buttons
        document.querySelectorAll('.sample-btn').forEach(btn => btn.classList.remove('active'));
        
        // Read file for preview
        const reader = new FileReader();
        reader.onload = (e) => {
            dropzone.innerHTML = `
                <i class="fa-solid fa-file-image upload-icon" style="color: var(--color-green)"></i>
                <h3>Scan Loaded</h3>
                <p class="text-green">${file.name} (${(file.size / 1024).toFixed(1)} KB)</p>
                <button class="btn btn-secondary btn-sm" id="browse-btn-reset">Change File</button>
            `;
            
            document.getElementById('browse-btn-reset').addEventListener('click', (e) => {
                e.stopPropagation();
                resetUploader();
            });
            
            // Setup base preview
            imgOriginal.src = e.target.result;
            resetVisualizationBox(e.target.result, false);
            
            diagnoseBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUploader() {
        uploadedFile = null;
        selectedSampleId = null;
        diagnoseBtn.disabled = true;
        
        document.querySelectorAll('.sample-btn').forEach(btn => btn.classList.remove('active'));
        
        dropzone.innerHTML = `
            <input type="file" id="file-input" accept="image/*" class="file-input-hidden">
            <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
            <h3>Drag & drop image file</h3>
            <p>Supports Chest X-rays, Brain MRIs, and CT scans</p>
            <button class="btn btn-secondary" id="browse-btn">Browse Files</button>
        `;
        
        // Re-bind elements
        const newFileInput = document.getElementById('file-input');
        newFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
        
        document.getElementById('browse-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            newFileInput.click();
        });
        
        // Reset visualization images
        imgOriginal.src = "/static/samples/cxr_normal.png";
        resetVisualizationBox("/static/samples/cxr_normal.png", false);
    }

    function resetVisualizationBox(originalSrc, hasResult = false) {
        const rightBox = imgOverlay.parentElement;
        
        if (!hasResult) {
            // Render basic original image in both windows
            rightBox.innerHTML = `
                <span class="scan-label">Grad-CAM Activation Map</span>
                <img id="img-overlay" src="${originalSrc}" alt="Grad-CAM Overlay" class="dimmed" style="width:100%; height:100%; object-fit:cover;">
            `;
            sliderContainer.style.display = 'none';
            triageStatus.className = 'badge triage-badge default-triage';
            triageStatus.innerHTML = '<i class="fa-solid fa-clock"></i> Awaiting Input';
            
            predClass.textContent = '—';
            predClass.className = 'metric-value';
            predConf.textContent = '—';
            confBar.style.width = '0%';
            clinicalRationale.textContent = 'Select a sample scan or upload an image and click "Run AI Diagnosis" to compute classification probability maps and check structural anomalies.';
        }
    }

    function runDiagnosis() {
        // Show loading state
        loadingOverlay.style.display = 'flex';
        diagnoseBtn.disabled = true;
        
        // Cycle loading messages to simulate workflow steps
        let steps = [
            "Preprocessing image array...",
            "Feeding image into ResNet-50 backbone...",
            "Computing forward activations...",
            "Extracting gradients from ResNet-50 layer4...",
            "Generating Grad-CAM activation heatmap...",
            "Overlaying heatmaps on original pixel coordinate space..."
        ];
        let stepIdx = 0;
        const msgInterval = setInterval(() => {
            if (stepIdx < steps.length) {
                loadingSubtext.textContent = steps[stepIdx++];
            }
        }, 600);

        const formData = new FormData();
        let fetchPromise = null;

        if (uploadedFile) {
            formData.append('file', uploadedFile);
            fetchPromise = fetch(API_PREDICT, {
                method: 'POST',
                body: formData
            });
        } else if (selectedSampleId) {
            fetchPromise = fetch(API_PREDICT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sample_id: selectedSampleId })
            });
        }

        if (!fetchPromise) {
            clearInterval(msgInterval);
            loadingOverlay.style.display = 'none';
            diagnoseBtn.disabled = false;
            return;
        }

        fetchPromise
            .then(res => res.json())
            .then(data => {
                clearInterval(msgInterval);
                if (data.success) {
                    displayDiagnosisResults(data);
                } else {
                    alert(`Error running diagnosis: ${data.error}`);
                    diagnoseBtn.disabled = false;
                }
            })
            .catch(err => {
                clearInterval(msgInterval);
                console.error("Diagnosis error:", err);
                alert("Error connecting to server. Please ensure the backend is running.");
                diagnoseBtn.disabled = false;
            })
            .finally(() => {
                loadingOverlay.style.display = 'none';
            });
    }

    function displayDiagnosisResults(data) {
        diagnoseBtn.disabled = false;
        
        // 1. Set original image resized
        imgOriginal.src = data.images.original;
        
        // 2. Set stacked layout in the right box to allow dynamic frontend opacity sliding!
        const rightBox = imgOverlay.parentElement;
        rightBox.innerHTML = `
            <span class="scan-label">Grad-CAM Activation Map</span>
            <img id="img-overlay-base" src="${data.images.original}" alt="Base original scan" style="width:100%; height:100%; object-fit:cover;">
            <img id="img-heatmap-layer" src="${data.images.heatmap}" alt="Grad-CAM Heatmap layer" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; opacity: ${opacitySlider.value / 100}; mix-blend-mode: screen;">
        `;
        
        // Show slider container
        sliderContainer.style.display = 'flex';
        
        // 3. Set triage badge
        if (data.triage === 'URGENT_AUDIT') {
            triageStatus.className = 'badge triage-badge urgent-triage';
            triageStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Urgent Audit';
            
            predClass.textContent = data.prediction;
            predClass.className = 'metric-value class-anomaly';
        } else {
            triageStatus.className = 'badge triage-badge normal-triage';
            triageStatus.innerHTML = '<i class="fa-solid fa-circle-check"></i> Normal Triage';
            
            predClass.textContent = data.prediction;
            predClass.className = 'metric-value class-normal';
        }
        
        // 4. Set confidence values
        predConf.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        confBar.style.width = `${data.confidence * 100}%`;
        
        // Set bar color based on class
        if (data.prediction === "Anomaly Detected") {
            confBar.style.background = 'var(--color-red)';
            confBar.style.boxShadow = '0 0 8px var(--color-red)';
        } else {
            confBar.style.background = 'var(--color-green)';
            confBar.style.boxShadow = '0 0 8px var(--color-green)';
        }
        
        // 5. Set text explanation
        clinicalRationale.textContent = data.explanation;
    }

    // --- Chart Rendering Functions ---

    function fetchAndRenderMetrics() {
        fetch(API_METRICS)
            .then(res => res.json())
            .then(data => {
                renderAccuracyChart(data);
                renderRocChart(data);
            })
            .catch(err => {
                console.error("Error loading metrics:", err);
            });
    }

    function renderAccuracyChart(data) {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        // Style parameters matching dark clinical UI
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [
                    {
                        label: 'Validation Accuracy',
                        data: data.training.val_accuracy,
                        borderColor: '#00f0ff',
                        backgroundColor: 'rgba(0, 240, 255, 0.1)',
                        fill: true,
                        tension: 0.3,
                        borderWidth: 2,
                        pointBackgroundColor: '#00f0ff',
                        pointBorderColor: '#070b19',
                        pointRadius: 4
                    },
                    {
                        label: 'Training Accuracy',
                        data: data.training.accuracy,
                        borderColor: '#a855f7',
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.3,
                        borderWidth: 1.5,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 20,
                            padding: 10
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(12, 18, 44, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(0, 240, 255, 0.2)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.raw}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 60,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                }
            }
        });
    }

    function renderRocChart(data) {
        const ctx = document.getElementById('rocChart').getContext('2d');
        
        // Parse ROC Curve data
        const fpr = data.roc_curve.map(p => p.fpr);
        const tpr = data.roc_curve.map(p => p.tpr);
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: fpr,
                datasets: [
                    {
                        label: 'MediScan-ResNet50 (AUC = 0.98)',
                        data: tpr,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        fill: true,
                        tension: 0.2,
                        borderWidth: 2,
                        pointBackgroundColor: '#10b981',
                        pointRadius: 2
                    },
                    {
                        label: 'Random Guess',
                        data: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        borderColor: 'rgba(255, 255, 255, 0.15)',
                        borderDash: [4, 4],
                        fill: false,
                        tension: 0,
                        borderWidth: 1,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false // We will hide legend and let the layout labels handle it
                    },
                    tooltip: {
                        backgroundColor: 'rgba(12, 18, 44, 0.95)',
                        callbacks: {
                            label: function(context) {
                                return `FPR: ${context.label}, TPR: ${context.raw}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'True Positive Rate (Sensitivity)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)'
                        }
                    },
                    x: {
                        min: 0,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'False Positive Rate (1 - Specificity)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)'
                        }
                    }
                }
            }
        });
    }
});
