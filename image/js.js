
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const progressBar = document.getElementById('progress-bar');
const slideNumDisplay = document.getElementById('slide-num');
let polynomialTablePage = 1;
let sourceDataTablePage = 1;

function renderLiveObjectTable(tableId, rows, columns, emptyMessage) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const tableBody = table.tBodies[0];
    tableBody.replaceChildren();
    rows.forEach(values => {
        const row = document.createElement('tr');
        columns.forEach(column => {
            const cell = document.createElement('td');
            cell.textContent = values[column] ?? '—';
            row.appendChild(cell);
        });
        tableBody.appendChild(row);
    });

    if (!rows.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = columns.length;
        cell.textContent = emptyMessage;
        row.appendChild(cell);
        tableBody.appendChild(row);
    }
}

function renderLivePlotlyChart(chartId, specification) {
    const chart = document.getElementById(chartId);
    if (!chart || !window.Plotly || !specification?.data) return;

    chart.hidden = false;
    const layout = {
        ...specification.layout,
        autosize: true,
        width: undefined,
    };
    Plotly.react(chart, specification.data, layout, {
        responsive: true,
        displaylogo: false,
        displayModeBar: true,
        scrollZoom: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: chartId,
            scale: 2,
        },
    });
}

function resizeVisiblePlotlyCharts(scope = document) {
    if (!window.Plotly) return;
    scope.querySelectorAll?.('.live-interactive-chart.js-plotly-plot').forEach(chart => {
        if (chart.getClientRects().length) Plotly.Plots.resize(chart);
    });
}

function renderPresentationBaseline(type) {
    const results = window.PRESENTATION_RESULTS || {};
    const summaries = results.baseline_summaries || {};
    const selectedType = summaries[type] ? type : (summaries.corrected ? 'corrected' : 'original');
    const summary = summaries[selectedType] || {};

    document.querySelectorAll('[data-baseline-card]').forEach(tab => {
        const isActive = tab.dataset.baselineCard === selectedType;
        tab.classList.toggle('is-active', isActive);
        tab.setAttribute('aria-selected', String(isActive));
        tab.tabIndex = isActive ? 0 : -1;
    });
    document.querySelectorAll('[data-baseline-key]').forEach(element => {
        element.textContent = summary[element.dataset.baselineKey] ?? '—';
    });
}

function renderPolynomialTable() {
    const results = window.PRESENTATION_RESULTS || {};
    const table = document.getElementById('live-polynomial-table');
    const pageSizeSelect = document.getElementById('polynomial-page-size');
    if (!table || !pageSizeSelect) return;

    const years = results.future_years || [];
    const rows = results.polynomial_table_rows || [];
    const pageSize = Number(pageSizeSelect.value);
    const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
    polynomialTablePage = Math.min(Math.max(polynomialTablePage, 1), pageCount);

    const headerRow = table.tHead.rows[0];
    headerRow.replaceChildren();
    ['ouName', 'deName', 'cocName', 'sexName', ...years].forEach(column => {
        const header = document.createElement('th');
        header.textContent = column;
        headerRow.appendChild(header);
    });

    const tableBody = table.tBodies[0];
    tableBody.replaceChildren();
    const firstRow = (polynomialTablePage - 1) * pageSize;
    rows.slice(firstRow, firstRow + pageSize).forEach(values => {
        const row = document.createElement('tr');
        values.forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.appendChild(cell);
        });
        tableBody.appendChild(row);
    });

    if (!rows.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 4 + years.length;
        cell.textContent = 'ບໍ່ມີຂໍ້ມູນສຳລັບຕົວກອງ (filter) ທີ່ເລືອກ';
        row.appendChild(cell);
        tableBody.appendChild(row);
    }

    const lastRow = Math.min(firstRow + pageSize, rows.length);
    document.getElementById('polynomial-row-status').textContent = rows.length
        ? `${(firstRow + 1).toLocaleString()}–${lastRow.toLocaleString()} ຈາກ ${rows.length.toLocaleString()} ແຖວ (rows)`
        : '0 ແຖວ (rows)';
    document.getElementById('polynomial-page-status').textContent =
        `ໜ້າ ${polynomialTablePage.toLocaleString()} ຈາກ ${pageCount.toLocaleString()} (Page ${polynomialTablePage.toLocaleString()} of ${pageCount.toLocaleString()})`;
    document.getElementById('polynomial-first-page').disabled = polynomialTablePage === 1;
    document.getElementById('polynomial-previous-page').disabled = polynomialTablePage === 1;
    document.getElementById('polynomial-next-page').disabled = polynomialTablePage === pageCount;
    document.getElementById('polynomial-last-page').disabled = polynomialTablePage === pageCount;
}

function renderSourceDataTable() {
    const results = window.PRESENTATION_RESULTS || {};
    const table = document.getElementById('live-source-data-table');
    const pageSizeSelect = document.getElementById('source-data-page-size');
    if (!table || !pageSizeSelect) return;

    const columns = results.source_data_columns || [];
    const rows = results.source_data_rows || [];
    const pageSize = Number(pageSizeSelect.value);
    const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
    sourceDataTablePage = Math.min(Math.max(sourceDataTablePage, 1), pageCount);

    const headerRow = table.tHead.rows[0];
    headerRow.replaceChildren();
    columns.forEach(column => {
        const header = document.createElement('th');
        header.textContent = column;
        headerRow.appendChild(header);
    });

    const tableBody = table.tBodies[0];
    tableBody.replaceChildren();
    const firstRow = (sourceDataTablePage - 1) * pageSize;
    rows.slice(firstRow, firstRow + pageSize).forEach(values => {
        const row = document.createElement('tr');
        values.forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value ?? '';
            row.appendChild(cell);
        });
        tableBody.appendChild(row);
    });

    if (!columns.length || !rows.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = Math.max(1, columns.length);
        cell.textContent = 'ບໍ່ພົບຂໍ້ມູນໃນ data/dataDistrict.csv (No CSV data found)';
        row.appendChild(cell);
        tableBody.appendChild(row);
    }

    const lastRow = Math.min(firstRow + pageSize, rows.length);
    document.getElementById('source-data-row-status').textContent = rows.length
        ? `${(firstRow + 1).toLocaleString()}–${lastRow.toLocaleString()} ຈາກ ${rows.length.toLocaleString()} ແຖວ (rows)`
        : '0 ແຖວ (rows)';
    document.getElementById('source-data-page-status').textContent =
        `ໜ້າ ${sourceDataTablePage.toLocaleString()} ຈາກ ${pageCount.toLocaleString()} (Page ${sourceDataTablePage.toLocaleString()} of ${pageCount.toLocaleString()})`;
    document.getElementById('source-data-first-page').disabled = sourceDataTablePage === 1;
    document.getElementById('source-data-previous-page').disabled = sourceDataTablePage === 1;
    document.getElementById('source-data-next-page').disabled = sourceDataTablePage === pageCount;
    document.getElementById('source-data-last-page').disabled = sourceDataTablePage === pageCount;
}

function applyLiveResults() {
    const results = window.PRESENTATION_RESULTS || {};
    document.querySelectorAll('[data-result-key]').forEach(element => {
        const value = results[element.dataset.resultKey];
        if (value !== undefined && value !== null) {
            element.textContent = value;
        }
    });
    document.querySelectorAll('[data-result-src-key]').forEach(element => {
        const source = results[element.dataset.resultSrcKey];
        if (source) {
            element.src = source;
        }
    });
    document.querySelectorAll('[data-model-row]').forEach(row => {
        row.classList.toggle('live-best-model', row.dataset.modelRow === results.best_model);
    });
    renderPresentationBaseline(results.forecast_input_type || 'corrected');

    const futureTable = document.getElementById('live-future-table');
    const years = results.future_years || [];
    const models = results.future_table_models || [];
    if (futureTable && years.length && models.length) {
        const headerRow = futureTable.tHead.rows[0];
        headerRow.replaceChildren();
        const modelHeader = document.createElement('th');
        modelHeader.textContent = 'ແບບຈຳລອງ (Models)';
        headerRow.appendChild(modelHeader);
        years.forEach(year => {
            const yearHeader = document.createElement('th');
            yearHeader.textContent = year;
            headerRow.appendChild(yearHeader);
        });

        const tableBody = futureTable.tBodies[0];
        tableBody.replaceChildren();
        models.forEach(model => {
            const row = document.createElement('tr');
            row.classList.toggle('live-best-model', model.name === results.best_model);
            const modelCell = document.createElement('td');
            modelCell.textContent = model.name;
            row.appendChild(modelCell);
            model.values.forEach(value => {
                const valueCell = document.createElement('td');
                valueCell.textContent = value;
                row.appendChild(valueCell);
            });
            tableBody.appendChild(row);
        });
    }

    renderLiveObjectTable(
        'live-correction-audit-table',
        results.correction_audit_rows || [],
        ['district', 'age_group', 'sex', 'year', 'original', 'corrected', 'method'],
        'ບໍ່ມີຄ່າທີ່ຜ່ານກົດການແກ້ໄຂແບບຮັດກຸມ (conservative correction rule) ສຳລັບຕົວກອງ (filter) ນີ້'
    );
    renderLiveObjectTable(
        'live-correction-impact-table',
        results.correction_impact_rows || [],
        ['model', 'year', 'original', 'corrected', 'difference', 'change'],
        'ບໍ່ມີຜົນກະທົບການຄາດຄະເນ (Forecast impact) ສຳລັບຕົວກອງ (filter) ນີ້'
    );
    renderLiveObjectTable(
        'live-validation-table',
        results.validation_rows || [],
        ['model', 'mae', 'rmse', 'mape', 'r2'],
        'ລໍຖ້າຜົນການກວດສອບຕາມເວລາ (time-based validation) ຈາກສ່ວນສາທິດ Streamlit (Demo)…'
    );
    renderLivePlotlyChart(
        'live-outlier-evidence-plotly',
        results.outlier_evidence_plotly
    );
    renderLivePlotlyChart(
        'live-correction-comparison-plotly',
        results.correction_comparison_plotly
    );
    renderLivePlotlyChart(
        'live-validation-plotly',
        results.validation_plotly
    );
    renderLivePlotlyChart(
        'live-forecast-plotly',
        results.forecast_plotly
    );
    renderLivePlotlyChart(
        'live-actual-predicted-plotly',
        results.actual_predicted_plotly
    );

    polynomialTablePage = 1;
    renderPolynomialTable();
    sourceDataTablePage = 1;
    renderSourceDataTable();
}

document.getElementById('source-data-page-size')?.addEventListener('change', () => {
    sourceDataTablePage = 1;
    renderSourceDataTable();
});
document.getElementById('source-data-first-page')?.addEventListener('click', () => {
    sourceDataTablePage = 1;
    renderSourceDataTable();
});
document.getElementById('source-data-previous-page')?.addEventListener('click', () => {
    sourceDataTablePage -= 1;
    renderSourceDataTable();
});
document.getElementById('source-data-next-page')?.addEventListener('click', () => {
    sourceDataTablePage += 1;
    renderSourceDataTable();
});
document.getElementById('source-data-last-page')?.addEventListener('click', () => {
    const rows = (window.PRESENTATION_RESULTS || {}).source_data_rows || [];
    const pageSize = Number(document.getElementById('source-data-page-size').value);
    sourceDataTablePage = Math.max(1, Math.ceil(rows.length / pageSize));
    renderSourceDataTable();
});

document.getElementById('polynomial-page-size')?.addEventListener('change', () => {
    polynomialTablePage = 1;
    renderPolynomialTable();
});
document.querySelectorAll('[data-baseline-card]').forEach(tab => {
    tab.addEventListener('click', () => {
        renderPresentationBaseline(tab.dataset.baselineCard);
    });
});
document.getElementById('polynomial-first-page')?.addEventListener('click', () => {
    polynomialTablePage = 1;
    renderPolynomialTable();
});
document.getElementById('polynomial-previous-page')?.addEventListener('click', () => {
    polynomialTablePage -= 1;
    renderPolynomialTable();
});
document.getElementById('polynomial-next-page')?.addEventListener('click', () => {
    polynomialTablePage += 1;
    renderPolynomialTable();
});
document.getElementById('polynomial-last-page')?.addEventListener('click', () => {
    const rows = (window.PRESENTATION_RESULTS || {}).polynomial_table_rows || [];
    const pageSize = Number(document.getElementById('polynomial-page-size').value);
    polynomialTablePage = Math.max(1, Math.ceil(rows.length / pageSize));
    renderPolynomialTable();
});

function showSlide(index, direction = "next") {

    // Remove active + animation classes
    slides.forEach(s => {
        s.classList.remove(
            'active',
            'slide-next',
            'slide-back'
        );
    });

    // Update current slide index
    if (index >= slides.length) currentSlide = 0;
    else if (index < 0) currentSlide = slides.length - 1;
    else currentSlide = index;

    // Show current slide
    const current = slides[currentSlide];
    current.classList.add('active');
    current.scrollTop = 0;

    // Add animation direction
    if (direction === "next") {
        current.classList.add('slide-next');
    } else {
        current.classList.add('slide-back');
    }

    // Update Progress Bar
    const progress = ((currentSlide + 1) / slides.length) * 100;
    progressBar.style.width = progress + '%';

    // Update Slide Number
    slideNumDisplay.innerText =
        `${currentSlide + 1} / ${slides.length}`;
    requestAnimationFrame(() => resizeVisiblePlotlyCharts(current));
}

function changeSlide(step) {
    if (step > 0) {
        showSlide(currentSlide + step, "next");
    } else {
        showSlide(currentSlide + step, "back");
    }
}

// Keyboard support
document.addEventListener('keydown', (e) => {
    if (document.getElementById('slide-overview')?.open) return;
    if (e.key === "ArrowRight") changeSlide(1);
    if (e.key === "ArrowLeft") changeSlide(-1);
});

function buildSlideOverview() {
    const grid = document.getElementById('slide-overview-grid');
    if (!grid || grid.childElementCount) return;

    slides.forEach((slide, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'slide-overview-card';
        button.setAttribute('aria-label', `ໄປຫາສະໄລດ໌ ${index + 1} (Go to slide ${index + 1})`);
        button.addEventListener('click', () => selectOverviewSlide(index));

        const frame = document.createElement('span');
        frame.className = 'slide-overview-frame';
        frame.setAttribute('aria-hidden', 'true');

        const preview = slide.cloneNode(true);
        preview.classList.remove('active', 'slide-next', 'slide-back');
        preview.classList.add('slide-overview-preview');
        preview.removeAttribute('id');
        preview.querySelectorAll('[id]').forEach(element => element.removeAttribute('id'));
        preview.querySelectorAll('button, input, select, textarea, a, summary, [tabindex]')
            .forEach(element => element.setAttribute('tabindex', '-1'));
        frame.appendChild(preview);

        const label = document.createElement('span');
        label.className = 'slide-overview-label';
        label.textContent = `ສະໄລດ໌ ${index + 1} (Slide ${index + 1})`;

        button.append(frame, label);
        grid.appendChild(button);
    });
}

function openSlideOverview() {
    const dialog = document.getElementById('slide-overview');
    if (!dialog) return;
    buildSlideOverview();
    dialog.showModal();

    const cards = dialog.querySelectorAll('.slide-overview-card');
    cards.forEach((card, index) => card.classList.toggle('is-current', index === currentSlide));
    cards[currentSlide]?.scrollIntoView({ block: 'center', inline: 'center' });
    cards[currentSlide]?.focus({ preventScroll: true });
}

function closeSlideOverview() {
    document.getElementById('slide-overview')?.close();
}

function selectOverviewSlide(index) {
    closeSlideOverview();
    showSlide(index, index >= currentSlide ? 'next' : 'back');
}

function togglePresentationFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen?.();
    } else {
        document.exitFullscreen?.();
    }
}

document.addEventListener('fullscreenchange', () => {
    const button = document.getElementById('presentation-fullscreen');
    if (!button) return;
    button.innerText = document.fullscreenElement ? '✕ ອອກຈາກເຕັມຈໍ' : '⛶ ເຕັມຈໍ';
    requestAnimationFrame(() => resizeVisiblePlotlyCharts());
});

document.querySelectorAll('details').forEach(details => {
    details.addEventListener('toggle', () => {
        if (details.open) requestAnimationFrame(() => resizeVisiblePlotlyCharts(details));
    });
});

// First slide
applyLiveResults();
const requestedSlide = Number(new URLSearchParams(window.location.search).get('slide'));
const initialSlide = Number.isInteger(requestedSlide) && requestedSlide > 0
    ? requestedSlide - 1
    : 0;
showSlide(initialSlide, "next");


// MINIMIZE
function toggleMinimize(button) {

    const panel =
        button.closest(".st-panel");

    // panel.classList.toggle("minimized");
    if (panel.classList.toggle("minimized")) {
        button.innerHTML = "+";
    } else {
        button.innerHTML = "─";
    }
}

// FULLSCREEN
function toggleFullscreen(button) {

    const panel =
        button.closest(".st-panel");

    panel.classList.toggle("fullscreen");

    // CHANGE ICON
    if (panel.classList.contains("fullscreen")) {
        // button.innerHTML = "✕";
        // button.innerHTML = "╬";
        button.innerHTML = "🟠";

    } else {
        // button.innerHTML = "⛶";
        button.innerHTML = "🟢";

    }
}
