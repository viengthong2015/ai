
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const progressBar = document.getElementById('progress-bar');
const slideNumDisplay = document.getElementById('slide-num');

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

// First slide
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