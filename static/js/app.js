/**
 * CrediPulse AI - Frontend Application Controller
 * Manages loan simulator sliders, real-time ML inference, Chart.js benchmarks,
 * dataset exploration, and batch portfolio underwriting.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Initializations
    initTabs();
    initSliders();
    initPresetButtons();
    initPredictionButton();
    loadBenchmarks();
    loadDatasetRecords(0);
    initDatasetControls();
    initBatchProcessing();
    initApiTabs();
    
    // Run initial prediction on default values
    runPrediction();
});

// State Store
const state = {
    datasetOffset: 0,
    datasetLimit: 25,
    datasetFilter: "",
    datasetSearch: "",
    batchResults: []
};

// ==========================================
// Tab Switching
// ==========================================
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const cat = btn.dataset.category;
            document.querySelectorAll(".category-group").forEach(group => {
                group.classList.add("hidden");
            });
            const activeGroup = document.getElementById(`cat-${cat}`);
            if (activeGroup) activeGroup.classList.remove("hidden");
        });
    });
}

// ==========================================
// Sliders & Input Syncing
// ==========================================
function initSliders() {
    const bindings = [
        { id: "inp-income", valId: "val-income", format: v => `$${Number(v).toLocaleString()}` },
        { id: "inp-age", valId: "val-age", format: v => `${v} yrs` },
        { id: "inp-emp-length", valId: "val-emp-length", format: v => `${v} yrs` },
        { id: "inp-loan-amnt", valId: "val-loan-amnt", format: v => `$${Number(v).toLocaleString()}` },
        { id: "inp-int-rate", valId: "val-int-rate", format: v => `${parseFloat(v).toFixed(2)}%` },
        { id: "inp-dti", valId: "val-dti", format: v => `${(parseFloat(v) * 100).toFixed(1)}%` },
        { id: "inp-utilization", valId: "val-utilization", format: v => `${(parseFloat(v) * 100).toFixed(1)}%` },
        { id: "inp-cred-hist", valId: "val-cred-hist", format: v => `${v} yrs` },
        { id: "inp-open-acc", valId: "val-open-acc", format: v => `${v} lines` }
    ];

    bindings.forEach(({ id, valId, format }) => {
        const input = document.getElementById(id);
        const display = document.getElementById(valId);
        if (input && display) {
            input.addEventListener("input", () => {
                display.textContent = format(input.value);
            });
        }
    });

    // Auto-update interest rate suggestion on loan grade change
    const gradeSelect = document.getElementById("inp-loan-grade");
    const intRateInput = document.getElementById("inp-int-rate");
    const intRateDisplay = document.getElementById("val-int-rate");

    if (gradeSelect && intRateInput && intRateDisplay) {
        const gradeRates = { A: 7.2, B: 10.5, C: 13.8, D: 16.9, E: 20.1, F: 23.4, G: 26.5 };
        gradeSelect.addEventListener("change", () => {
            const suggested = gradeRates[gradeSelect.value] || 12.0;
            intRateInput.value = suggested;
            intRateDisplay.textContent = `${suggested.toFixed(2)}%`;
        });
    }
}

function getApplicantDataFromForm() {
    return {
        person_age: parseFloat(document.getElementById("inp-age").value),
        person_income: parseFloat(document.getElementById("inp-income").value),
        person_emp_length: parseFloat(document.getElementById("inp-emp-length").value),
        person_home_ownership: document.getElementById("inp-home-ownership").value,
        loan_amnt: parseFloat(document.getElementById("inp-loan-amnt").value),
        loan_intent: document.getElementById("inp-loan-intent").value,
        loan_grade: document.getElementById("inp-loan-grade").value,
        loan_int_rate: parseFloat(document.getElementById("inp-int-rate").value),
        debt_to_income_ratio: parseFloat(document.getElementById("inp-dti").value),
        revolving_utilization: parseFloat(document.getElementById("inp-utilization").value),
        cb_person_cred_hist_length: parseFloat(document.getElementById("inp-cred-hist").value),
        num_open_accounts: parseFloat(document.getElementById("inp-open-acc").value),
        cb_person_default_on_file: document.getElementById("inp-default-on-file").value
    };
}

function populateFormFromApplicant(app) {
    if (!app) return;
    
    const setVal = (id, val, format) => {
        const el = document.getElementById(id);
        if (el && val !== undefined) {
            el.value = val;
            if (format) {
                const disp = document.getElementById(id.replace("inp-", "val-"));
                if (disp) disp.textContent = format(val);
            }
        }
    };

    setVal("inp-income", app.person_income, v => `$${Number(v).toLocaleString()}`);
    setVal("inp-age", app.person_age, v => `${v} yrs`);
    setVal("inp-emp-length", app.person_emp_length, v => `${v} yrs`);
    setVal("inp-home-ownership", app.person_home_ownership);
    setVal("inp-loan-amnt", app.loan_amnt, v => `$${Number(v).toLocaleString()}`);
    setVal("inp-loan-intent", app.loan_intent);
    setVal("inp-loan-grade", app.loan_grade);
    setVal("inp-int-rate", app.loan_int_rate, v => `${parseFloat(v).toFixed(2)}%`);
    setVal("inp-dti", app.debt_to_income_ratio, v => `${(parseFloat(v) * 100).toFixed(1)}%`);
    setVal("inp-utilization", app.revolving_utilization, v => `${(parseFloat(v) * 100).toFixed(1)}%`);
    setVal("inp-cred-hist", app.cb_person_cred_hist_length, v => `${v} yrs`);
    setVal("inp-open-acc", app.num_open_accounts, v => `${v} lines`);
    setVal("inp-default-on-file", app.cb_person_default_on_file);
}

// ==========================================
// Presets Loader
// ==========================================
function initPresetButtons() {
    const presets = ["prime", "subprime", "borderline", "random"];
    presets.forEach(p => {
        const btn = document.getElementById(`btn-load-${p}`);
        if (btn) {
            btn.addEventListener("click", async () => {
                btn.style.opacity = "0.6";
                try {
                    const res = await fetch(`/api/sample/${p}`);
                    const data = await res.json();
                    if (data.status === "success" && data.data.applicant) {
                        populateFormFromApplicant(data.data.applicant);
                        runPrediction();
                    }
                } catch (e) {
                    console.error("Failed to load preset:", e);
                } finally {
                    btn.style.opacity = "1";
                }
            });
        }
    });

    const resetBtn = document.getElementById("btn-reset-baseline");
    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            populateFormFromApplicant({
                person_age: 32,
                person_income: 65000,
                person_emp_length: 5,
                person_home_ownership: "MORTGAGE",
                loan_amnt: 12000,
                loan_intent: "DEBTCONSOLIDATION",
                loan_grade: "B",
                loan_int_rate: 10.5,
                debt_to_income_ratio: 0.25,
                revolving_utilization: 0.32,
                cb_person_cred_hist_length: 8,
                num_open_accounts: 8,
                cb_person_default_on_file: "N"
            });
            runPrediction();
        });
    }
}

// ==========================================
// Real-time Prediction Execution
// ==========================================
function initPredictionButton() {
    const btn = document.getElementById("btn-run-prediction");
    if (btn) {
        btn.addEventListener("click", () => runPrediction());
    }
}

async function runPrediction() {
    const btn = document.getElementById("btn-run-prediction");
    if (btn) btn.classList.add("loading");

    const payload = { applicant: getApplicantDataFromForm() };

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.status === "success") {
            renderPredictionResults(data.result);
        }
    } catch (e) {
        console.error("Prediction error:", e);
    } finally {
        if (btn) btn.classList.remove("loading");
    }
}

function renderPredictionResults(result) {
    const { consensus, models, risk_drivers } = result;

    // Consensus FICO Gauge & Verdict
    const ficoEl = document.getElementById("gauge-fico-score");
    const gaugeBar = document.getElementById("gauge-bar");
    const verdictBadge = document.getElementById("diagnosis-badge");
    const verdictIcon = document.getElementById("verdict-icon");
    const verdictText = document.getElementById("verdict-text");
    const riskTag = document.getElementById("risk-level-tag");
    const verdictSummary = document.getElementById("verdict-summary");
    const agreementBadge = document.getElementById("consensus-agreement");

    if (ficoEl) ficoEl.textContent = consensus.credit_score;
    if (agreementBadge) agreementBadge.textContent = consensus.agreement;
    if (verdictSummary) verdictSummary.textContent = consensus.recommendation;

    // FICO Gauge Animation: Circumference = 2 * PI * 68 = ~427
    const maxScore = 850;
    const minScore = 300;
    const scorePct = Math.max(0, Math.min(1, (consensus.credit_score - minScore) / (maxScore - minScore)));
    const totalCirc = 427;
    const offset = totalCirc * (1 - scorePct);

    if (gaugeBar) {
        gaugeBar.style.strokeDashoffset = offset;
        gaugeBar.style.stroke = consensus.is_default ? "#f43f5e" : (consensus.credit_score >= 750 ? "#10b981" : "#38bdf8");
    }

    if (consensus.is_default) {
        verdictBadge.className = "diagnosis-badge badge-rejected";
        verdictIcon.textContent = "⚠️";
        verdictText.textContent = "REJECTED (HIGH RISK)";
    } else {
        verdictBadge.className = "diagnosis-badge badge-approved";
        verdictIcon.textContent = "✅";
        verdictText.textContent = "APPROVED";
    }

    if (riskTag) {
        riskTag.textContent = consensus.risk_tier;
        riskTag.className = `risk-level-tag tag-${consensus.risk_badge}`;
    }

    // Probability Track
    const valApprove = document.getElementById("val-approve-prob");
    const valDefault = document.getElementById("val-default-prob");
    const barApprove = document.getElementById("bar-approve");
    const barDefault = document.getElementById("bar-default");

    if (valApprove) valApprove.textContent = `${consensus.avg_approve_prob}%`;
    if (valDefault) valDefault.textContent = `${consensus.avg_default_prob}%`;
    if (barApprove) barApprove.style.width = `${consensus.avg_approve_prob}%`;
    if (barDefault) barDefault.style.width = `${consensus.avg_default_prob}%`;

    // Tri-Model Subcards
    const updateModelCard = (prefix, modelData) => {
        const badge = document.getElementById(`badge-${prefix}`);
        const prob = document.getElementById(`prob-${prefix}`);
        const prog = document.getElementById(`prog-${prefix}`);

        if (badge && modelData) {
            badge.textContent = modelData.is_default ? "High Risk" : "Approved";
            badge.className = `subcard-badge ${modelData.is_default ? "badge-rejected" : "badge-approved"}`;
        }
        if (prob && modelData) {
            prob.textContent = `${modelData.is_default ? modelData.prob_default : modelData.prob_approved}%`;
        }
        if (prog && modelData) {
            prog.style.width = `${modelData.is_default ? modelData.prob_default : modelData.prob_approved}%`;
            prog.className = `mini-progress-fill ${modelData.is_default ? "fill-risk" : "fill-good"}`;
        }
    };

    updateModelCard("gb", models.GradientBoosting);
    updateModelCard("rf", models.RandomForest);
    updateModelCard("lr", models.LogisticRegression);

    // Risk Drivers List
    const riskContainer = document.getElementById("risk-drivers-container");
    if (riskContainer) {
        riskContainer.innerHTML = "";
        risk_drivers.forEach(driver => {
            const item = document.createElement("div");
            item.className = `risk-item ${driver.severity}`;
            const icon = driver.severity === "critical" ? "🚨" : (driver.severity === "high" ? "⚠️" : (driver.severity === "medium" ? "📊" : "🛡️"));
            item.innerHTML = `
                <span class="risk-icon">${icon}</span>
                <div class="risk-details">
                    <div class="risk-title">${driver.factor}</div>
                    <div class="risk-desc">${driver.detail}</div>
                </div>
            `;
            riskContainer.appendChild(item);
        });
    }
}

// ==========================================
// Benchmarks & Chart.js Visualizations
// ==========================================
let comparisonChart = null;
let rocChart = null;
let featImpChart = null;

async function loadBenchmarks() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        if (data.status === "success") {
            renderModelComparisonChart(data.metrics);
            renderRocChart(data.roc_data);
            renderConfusionMatrices(data.confusion_matrices);
            renderMetricsTable(data.metrics);
            renderFeatureImportanceChart(data.feature_importances);
        }
    } catch (e) {
        console.error("Failed to load benchmarks:", e);
    }
}

function renderModelComparisonChart(metrics) {
    const ctx = document.getElementById("chart-model-comparison");
    if (!ctx) return;

    const labels = Object.keys(metrics);
    const accuracies = labels.map(l => (metrics[l].accuracy * 100).toFixed(2));
    const aucScores = labels.map(l => (metrics[l].roc_auc * 100).toFixed(2));
    const f1Scores = labels.map(l => (metrics[l].f1_score * 100).toFixed(2));

    if (comparisonChart) comparisonChart.destroy();

    comparisonChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Accuracy (%)",
                    data: accuracies,
                    backgroundColor: "rgba(16, 185, 129, 0.75)",
                    borderColor: "#10b981",
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: "ROC-AUC (%)",
                    data: aucScores,
                    backgroundColor: "rgba(56, 189, 248, 0.75)",
                    borderColor: "#38bdf8",
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: "F1-Score (%)",
                    data: f1Scores,
                    backgroundColor: "rgba(129, 140, 248, 0.75)",
                    borderColor: "#818cf8",
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans", size: 11 } } }
            },
            scales: {
                y: {
                    min: 70,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { weight: "600" } }
                }
            }
        }
    });
}

function renderRocChart(rocData) {
    const ctx = document.getElementById("chart-roc-curve");
    if (!ctx) return;

    const colors = {
        GradientBoosting: "#10b981",
        RandomForest: "#38bdf8",
        LogisticRegression: "#f59e0b"
    };

    const datasets = Object.keys(rocData).map(name => {
        const dataPoints = rocData[name].fpr.map((fpr, i) => ({
            x: fpr,
            y: rocData[name].tpr[i]
        }));
        return {
            label: `${name} (AUC: ${rocData[name].auc})`,
            data: dataPoints,
            borderColor: colors[name] || "#38bdf8",
            backgroundColor: "transparent",
            borderWidth: 2.2,
            tension: 0.2,
            pointRadius: 0
        };
    });

    // Add Chance Diagonal
    datasets.push({
        label: "Chance (AUC: 0.50)",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(255, 255, 255, 0.2)",
        borderDash: [5, 5],
        borderWidth: 1.5,
        pointRadius: 0
    });

    if (rocChart) rocChart.destroy();

    rocChart = new Chart(ctx, {
        type: "line",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans", size: 11 } } }
            },
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#64748b" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: "True Positive Rate (Sensitivity)", color: "#64748b" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                }
            }
        }
    });
}

function renderConfusionMatrices(cms) {
    const populateCm = (prefix, cmData) => {
        if (!cmData) return;
        const tn = document.getElementById(`cm-${prefix}-tn`);
        const fp = document.getElementById(`cm-${prefix}-fp`);
        const fn = document.getElementById(`cm-${prefix}-fn`);
        const tp = document.getElementById(`cm-${prefix}-tp`);

        if (tn) tn.textContent = cmData.good_as_good;
        if (fp) fp.textContent = cmData.good_as_default;
        if (fn) fn.textContent = cmData.default_as_good;
        if (tp) tp.textContent = cmData.default_as_default;
    };

    populateCm("gb", cms.GradientBoosting);
    populateCm("rf", cms.RandomForest);
    populateCm("lr", cms.LogisticRegression);
}

function renderMetricsTable(metrics) {
    const tbody = document.getElementById("metrics-tbody");
    if (!tbody) return;

    tbody.innerHTML = "";
    Object.keys(metrics).forEach(name => {
        const m = metrics[name];
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${(m.accuracy * 100).toFixed(2)}%</td>
            <td><strong style="color:#38bdf8;">${m.roc_auc.toFixed(4)}</strong></td>
            <td>${(m.precision * 100).toFixed(2)}%</td>
            <td>${(m.recall * 100).toFixed(2)}%</td>
            <td>${(m.f1_score * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderFeatureImportanceChart(featureImportances) {
    const ctx = document.getElementById("chart-feature-importance");
    if (!ctx) return;

    const list = featureImportances.GradientBoosting || featureImportances.RandomForest || [];
    const top10 = list.slice(0, 10);
    const labels = top10.map(item => item.feature);
    const vals = top10.map(item => item.importance);

    if (featImpChart) featImpChart.destroy();

    featImpChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Relative Importance",
                data: vals,
                backgroundColor: "rgba(56, 189, 248, 0.75)",
                borderColor: "#38bdf8",
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { size: 10, family: "JetBrains Mono" } }
                }
            }
        }
    });
}

// ==========================================
// Dataset Explorer
// ==========================================
function initDatasetControls() {
    const searchInput = document.getElementById("dataset-search");
    const filterSelect = document.getElementById("dataset-filter");
    const prevBtn = document.getElementById("btn-page-prev");
    const nextBtn = document.getElementById("btn-page-next");

    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            state.datasetSearch = e.target.value.toLowerCase();
            state.datasetOffset = 0;
            loadDatasetRecords(0);
        });
    }

    if (filterSelect) {
        filterSelect.addEventListener("change", (e) => {
            state.datasetFilter = e.target.value;
            state.datasetOffset = 0;
            loadDatasetRecords(0);
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            if (state.datasetOffset >= state.datasetLimit) {
                state.datasetOffset -= state.datasetLimit;
                loadDatasetRecords(state.datasetOffset);
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            state.datasetOffset += state.datasetLimit;
            loadDatasetRecords(state.datasetOffset);
        });
    }
}

async function loadDatasetRecords(offset = 0) {
    try {
        const url = `/api/dataset/records?limit=${state.datasetLimit}&offset=${offset}&filter=${encodeURIComponent(state.datasetFilter)}`;
        const res = await fetch(url);
        const data = await res.json();

        if (data.status === "success") {
            renderDatasetTable(data.data);
        }
    } catch (e) {
        console.error("Failed to load dataset records:", e);
    }
}

function renderDatasetTable(datasetData) {
    const tbody = document.getElementById("dataset-tbody");
    const pageInfo = document.getElementById("pagination-info");
    const pageCurrent = document.getElementById("page-current");
    const prevBtn = document.getElementById("btn-page-prev");
    const nextBtn = document.getElementById("btn-page-next");

    if (!tbody) return;
    tbody.innerHTML = "";

    const records = datasetData.records;
    records.forEach(rec => {
        const d = rec.data;
        const tr = document.createElement("tr");
        const statusBadge = rec.status === "Approved" ? '<span class="subcard-badge badge-approved">Approved</span>' : '<span class="subcard-badge badge-rejected">Default Risk</span>';

        tr.innerHTML = `
            <td><code>#APP-${1000 + rec.id}</code></td>
            <td>${statusBadge}</td>
            <td>${d.person_age || '-'}</td>
            <td>$${Number(d.person_income || 0).toLocaleString()}</td>
            <td>$${Number(d.loan_amnt || 0).toLocaleString()}</td>
            <td>${d.loan_int_rate || '-'}%</td>
            <td>${d.debt_to_income_ratio ? (d.debt_to_income_ratio * 100).toFixed(1) + '%' : '-'}</td>
            <td><strong>Grade ${d.loan_grade || '-'}</strong></td>
            <td>${d.revolving_utilization ? (d.revolving_utilization * 100).toFixed(1) + '%' : '-'}</td>
            <td><button class="btn btn-secondary btn-sm" onclick='loadApplicantRowIntoStudio(${JSON.stringify(d)})'>Load</button></td>
        `;
        tbody.appendChild(tr);
    });

    const start = datasetData.offset + 1;
    const end = Math.min(datasetData.offset + datasetData.limit, datasetData.total);
    if (pageInfo) pageInfo.textContent = `Showing ${start} to ${end} of ${datasetData.total} records`;
    if (pageCurrent) pageCurrent.textContent = Math.floor(datasetData.offset / datasetData.limit) + 1;

    if (prevBtn) prevBtn.disabled = datasetData.offset === 0;
    if (nextBtn) nextBtn.disabled = end >= datasetData.total;
}

window.loadApplicantRowIntoStudio = function(applicant) {
    populateFormFromApplicant(applicant);
    runPrediction();
    window.location.hash = "#simulator-studio";
};

// ==========================================
// Batch Underwriting Suite
// ==========================================
function initBatchProcessing() {
    const fileInput = document.getElementById("batch-file-input");
    const sampleBatchBtn = document.getElementById("btn-load-sample-batch");
    const exportBtn = document.getElementById("btn-export-batch");
    const dropzone = document.getElementById("csv-dropzone");

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                processBatchFile(e.target.files[0]);
            }
        });
    }

    if (dropzone) {
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "#10b981";
        });
        dropzone.addEventListener("dragleave", () => {
            dropzone.style.borderColor = "rgba(255,255,255,0.15)";
        });
        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "rgba(255,255,255,0.15)";
            if (e.dataTransfer.files.length > 0) {
                processBatchFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (sampleBatchBtn) {
        sampleBatchBtn.addEventListener("click", async () => {
            sampleBatchBtn.style.opacity = "0.6";
            try {
                // Fetch 20 records from dataset explorer
                const res = await fetch("/api/dataset/records?limit=20&offset=0");
                const data = await res.json();
                if (data.status === "success") {
                    const applicants = data.data.records.map(r => r.data);
                    const batchRes = await fetch("/api/batch-predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ applicants })
                    });
                    const batchData = await batchRes.json();
                    if (batchData.status === "success") {
                        renderBatchResults(batchData.data);
                    }
                }
            } catch (e) {
                console.error("Batch sample failed:", e);
            } finally {
                sampleBatchBtn.style.opacity = "1";
            }
        });
    }

    if (exportBtn) {
        exportBtn.addEventListener("click", () => exportBatchResultsCsv());
    }
}

async function processBatchFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/batch-predict", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.status === "success") {
            renderBatchResults(data.data);
        } else {
            alert("Batch error: " + data.message);
        }
    } catch (e) {
        console.error("Batch upload failed:", e);
    }
}

function renderBatchResults(batchData) {
    state.batchResults = batchData.results;

    document.getElementById("batch-total").textContent = batchData.total_processed;
    document.getElementById("batch-approved").textContent = batchData.approved_count;
    document.getElementById("batch-rejected").textContent = batchData.rejected_count;
    document.getElementById("batch-rate").textContent = `${batchData.approval_rate}%`;

    const exportBtn = document.getElementById("btn-export-batch");
    if (exportBtn) exportBtn.disabled = false;

    const wrapper = document.getElementById("batch-results-wrapper");
    const tbody = document.getElementById("batch-results-tbody");
    if (wrapper) wrapper.style.display = "block";
    if (tbody) {
        tbody.innerHTML = "";
        batchData.results.forEach(r => {
            const tr = document.createElement("tr");
            const decisionBadge = r.is_default ? '<span class="subcard-badge badge-rejected">Rejected</span>' : '<span class="subcard-badge badge-approved">Approved</span>';
            tr.innerHTML = `
                <td><code>${r.application_id}</code></td>
                <td>${decisionBadge}</td>
                <td><strong>${r.credit_score}</strong></td>
                <td>${r.risk_tier}</td>
                <td>${r.confidence}%</td>
                <td>${r.gradient_boosting}</td>
                <td>${r.random_forest}</td>
                <td>${r.logistic_regression}</td>
            `;
            tbody.appendChild(tr);
        });
    }
}

function exportBatchResultsCsv() {
    if (!state.batchResults || state.batchResults.length === 0) return;

    const headers = ["Application ID", "Decision", "FICO Score", "Risk Tier", "Confidence (%)", "Is Default", "Gradient Boosting", "Random Forest", "Logistic Regression"];
    const rows = state.batchResults.map(r => [
        r.application_id,
        r.decision,
        r.credit_score,
        `"${r.risk_tier}"`,
        r.confidence,
        r.is_default,
        r.gradient_boosting,
        r.random_forest,
        r.logistic_regression
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `CrediPulse_Underwriting_Report_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ==========================================
// API Documentation Playground
// ==========================================
function initApiTabs() {
    const tabs = document.querySelectorAll(".api-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const lang = tab.dataset.lang;
            document.querySelectorAll(".code-block").forEach(b => b.classList.remove("active"));
            const target = document.getElementById(`code-${lang}`);
            if (target) target.classList.add("active");
        });
    });
}
