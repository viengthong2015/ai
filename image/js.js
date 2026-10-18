
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const progressBar = document.getElementById('progress-bar');
const slideNumDisplay = document.getElementById('slide-num');

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
}

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
    if (e.key === "ArrowRight") changeSlide(1);
    if (e.key === "ArrowLeft") changeSlide(-1);
});

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
