// Generic dropzone handler
function setupDropzone(dropEl, inputEl, onPreview) {
  dropEl.addEventListener('click', () => inputEl.click());
  ['dragenter','dragover'].forEach(evt => {
    dropEl.addEventListener(evt, e => { e.preventDefault(); dropEl.classList.add('hover'); });
  });
  ['dragleave','drop'].forEach(evt => {
    dropEl.addEventListener(evt, e => { e.preventDefault(); dropEl.classList.remove('hover'); });
  });
  dropEl.addEventListener('drop', e => {
    const file = e.dataTransfer.files?.[0];
    if (file) {
      inputEl.files = e.dataTransfer.files;
      onPreview(file);
    }
  });
  inputEl.addEventListener('change', e => {
    const file = inputEl.files?.[0];
    if (file) onPreview(file);
  });
}

// Image preview
const imgDrop = document.getElementById('imgDrop');
const imgInput = document.getElementById('image_file');
const imgPrev = document.getElementById('imgPreview');
if (imgDrop && imgInput && imgPrev) {
  setupDropzone(imgDrop, imgInput, (file) => {
    const reader = new FileReader();
    reader.onload = e => {
      imgPrev.innerHTML = `<img src="${e.target.result}" alt="preview">`;
    };
    reader.readAsDataURL(file);
  });
}

// Audio preview
const audDrop = document.getElementById('audDrop');
const audInput = document.getElementById('audio_file');
const audPrev = document.getElementById('audPreview');
if (audDrop && audInput && audPrev) {
  setupDropzone(audDrop, audInput, (file) => {
    audPrev.innerHTML = `<div class="fileinfo">Selected: ${file.name} (${Math.round(file.size/1024)} KB)</div>`;
  });
}

// Reset forms after submit so user can upload same file again
const imgForm = document.getElementById('imgForm');
if (imgForm) {
  imgForm.addEventListener('submit', () => setTimeout(() => imgForm.reset(), 600));
}
const audForm = document.getElementById('audForm');
if (audForm) {
  audForm.addEventListener('submit', () => setTimeout(() => audForm.reset(), 600));
}
// --- Loader animation ---
const loader = document.getElementById("loader");

function showLoader() {
  if (loader) {
    loader.classList.remove("hidden");
    // safety auto-hide after 6 s
    setTimeout(() => loader.classList.add("hidden"), 6000);
  }
}

// Attach loader to all forms
document.querySelectorAll("form").forEach(f => {
  f.addEventListener("submit", showLoader);
});
