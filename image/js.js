
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const progressBar = document.getElementById('progress-bar');
const slideNumDisplay = document.getElementById('slide-num');
let polynomialTablePage = 1;

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
        cell.textContent = 'ບໍ່ມີຂໍ້ມູນສຳລັບ filter ທີ່ເລືອກ';
        row.appendChild(cell);
        tableBody.appendChild(row);
    }

    const lastRow = Math.min(firstRow + pageSize, rows.length);
    document.getElementById('polynomial-row-status').textContent = rows.length
        ? `${(firstRow + 1).toLocaleString()}–${lastRow.toLocaleString()} of ${rows.length.toLocaleString()} rows`
        : '0 rows';
    document.getElementById('polynomial-page-status').textContent =
        `Page ${polynomialTablePage.toLocaleString()} of ${pageCount.toLocaleString()}`;
    document.getElementById('polynomial-first-page').disabled = polynomialTablePage === 1;
    document.getElementById('polynomial-previous-page').disabled = polynomialTablePage === 1;
    document.getElementById('polynomial-next-page').disabled = polynomialTablePage === pageCount;
    document.getElementById('polynomial-last-page').disabled = polynomialTablePage === pageCount;
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

    polynomialTablePage = 1;
    renderPolynomialTable();
}

document.getElementById('polynomial-page-size')?.addEventListener('change', () => {
    polynomialTablePage = 1;
    renderPolynomialTable();
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
        button.setAttribute('aria-label', `Go to slide ${index + 1}`);
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
        label.textContent = `Slide ${index + 1}`;

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
});

// First slide
applyLiveResults();
showSlide(0, "next");


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
