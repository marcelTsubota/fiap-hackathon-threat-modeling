// ===========================================
// FIAP Threat Modeling — Frontend JS
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const previewImg = document.getElementById('preview-img');
    const fileName = document.getElementById('file-name');
    const submitBtn = document.getElementById('submit-btn');
    const form = document.getElementById('upload-form');

    if (!dropZone || !fileInput) return;

    // Drag and drop
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Por favor, selecione um arquivo de imagem (PNG, JPG, JPEG).');
            return;
        }

        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 10) {
            alert('Arquivo muito grande. Máximo: 10 MB.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            fileName.textContent = `${file.name} (${sizeMB.toFixed(1)} MB)`;
            filePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Enable submit
        submitBtn.disabled = false;
    }

    // Form submit loading state
    if (form) {
        form.addEventListener('submit', () => {
            const btnText = submitBtn.querySelector('.btn-text');
            const btnLoading = submitBtn.querySelector('.btn-loading');
            if (btnText) btnText.style.display = 'none';
            if (btnLoading) btnLoading.style.display = 'inline';
            submitBtn.disabled = true;
        });
    }
});
