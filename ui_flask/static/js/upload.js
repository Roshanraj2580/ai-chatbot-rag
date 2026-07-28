// Upload functionality

// Handle file selection
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    uploadFiles(files);
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById('uploadArea').classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    document.getElementById('uploadArea').classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();

    const uploadArea = document.getElementById('uploadArea');
    uploadArea.classList.remove('dragover');

    const files = Array.from(event.dataTransfer.files).filter(f => f.type === 'application/pdf');

    if (files.length === 0) {
        AppUtils.showToast('Please drop PDF files only', 'warning');
        return;
    }

    uploadFiles(files);
}

// Upload files
async function uploadFiles(files) {
    if (files.length === 0) return;

    const progressDiv = document.getElementById('uploadProgress');
    const resultsDiv = document.getElementById('uploadResults');

    // Clear previous results
    resultsDiv.innerHTML = '';

    // Show progress
    progressDiv.style.display = 'block';

    for (let i = 0; i < files.length; i++) {
        const file = files[i];

        // Update progress text
        document.getElementById('progressText').textContent =
            `Uploading ${i + 1} of ${files.length}: ${file.name}`;

        // Update progress bar
        const percent = ((i) / files.length) * 100;
        document.getElementById('progressBar').style.width = `${percent}%`;

        try {
            await uploadSingleFile(file);

            // Show success
            addResult('success', file.name, 'Successfully uploaded and processed');

        } catch (error) {
            console.error(`Error uploading ${file.name}:`, error);
            addResult('danger', file.name, `Error: ${error.message}`);
        }
    }

    // Complete
    document.getElementById('progressBar').style.width = '100%';
    document.getElementById('progressText').textContent = 'Upload complete!';

    setTimeout(() => {
        progressDiv.style.display = 'none';
        document.getElementById('progressBar').style.width = '0%';
    }, 2000);

    // Clear file input
    document.getElementById('fileInput').value = '';
}

// Upload single file
async function uploadSingleFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Upload failed');
    }

    const result = await response.json();

    if (result.error) {
        throw new Error(result.error);
    }

    return result;
}

// Add result to UI
function addResult(type, filename, message) {
    const resultsDiv = document.getElementById('uploadResults');

    const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
    const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';

    const alert = document.createElement('div');
    alert.className = `alert ${alertClass} alert-dismissible fade show`;
    alert.innerHTML = `
        <i class="fas ${icon}"></i>
        <strong>${filename}</strong>: ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    resultsDiv.appendChild(alert);

    // Show toast
    AppUtils.showToast(`${filename}: ${message}`, type === 'success' ? 'success' : 'danger');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Prevent default drag behaviors on document
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });
});
