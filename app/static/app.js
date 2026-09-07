let selectedFiles = [];
let currentDocId = null;

function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    
    if(tabId === 'history-tab') fetchHistory();
}

// File Drag & Drop Handling
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('hover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('hover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

function handleFiles(files) {
    selectedFiles = Array.from(files).slice(0, 10);
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = selectedFiles.map(f => `<div>📄 ${f.name} (${(f.size/1024).toFixed(1)} KB)</div>`).join('');
    document.getElementById('start-upload-btn').disabled = selectedFiles.length === 0;
}

// Upload and Process Batch
async function handleUploadAndProcess() {
    const formData = new FormData();
    selectedFiles.forEach((file, index) => {
        formData.append(`file${index + 1}`, file);
    });

    try {
        const uploadRes = await fetch('/documents/upload', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();
        
        if (uploadData.documents && uploadData.documents.length > 0) {
            for (const doc of uploadData.documents) {
                await fetch(`/documents/${doc.document_id}/process`, { method: 'POST' });
            }
            alert("Batch uploaded and processing started!");
            showTab('history-tab');
        }
    } catch (err) {
        alert("Upload failed: " + err.message);
    }
}

// Fetch Document History
async function fetchHistory() {
    const res = await fetch('/documents');
    const docs = await res.json();
    const tbody = document.getElementById('history-table-body');
    tbody.innerHTML = docs.map(d => `
        <tr>
            <td><code>${d.document_id}</code></td>
            <td>${d.filename}</td>
            <td><span class="badge">${d.status}</span></td>
            <td><button class="btn btn-primary" onclick="loadReview('${d.document_id}')">Review</button></td>
        </tr>
    `).join('');
}

// Load Document into Review Workspace
async function loadReview(docId) {
    currentDocId = docId;
    showTab('review-tab');
    
    document.getElementById('doc-viewer-img').src = `/documents/${docId}/image?annotated=true`;
    document.getElementById('review-doc-status').innerText = 'LOADING...';

    const res = await fetch(`/documents/${docId}/result`);
    const data = await res.json();

    document.getElementById('review-doc-status').innerText = data.status;

    // Populate Fields from result JSON
    const page = data.results && data.results[0] ? data.results[0] : {};
    const consensus = page.field_consensus || {};

    document.getElementById('field-invoice_number').value = consensus.invoice_number || '';
    document.getElementById('field-invoice_date').value = consensus.invoice_date || '';
    document.getElementById('field-vendor_name').value = consensus.vendor_name || '';
    document.getElementById('field-tax_id').value = consensus.tax_id || '';
    document.getElementById('field-subtotal').value = consensus.subtotal || 0;
    document.getElementById('field-tax').value = consensus.tax || 0;
    document.getElementById('field-grand_total').value = consensus.grand_total || 0;

    // Quality Report
    const quality = page.quality_report || {};
    document.getElementById('quality-label').innerText = quality.quality_label || 'Good';
    document.getElementById('recommended-profile').innerText = quality.recommended_profile || 'BASIC';

    // Populate Line Items
    const lineItems = (page.line_items && Object.values(page.line_items)[0]) || [];
    const tbody = document.getElementById('line-items-body');
    tbody.innerHTML = lineItems.map(item => `
        <tr>
            <td><input type="text" value="${item.description}"></td>
            <td><input type="number" value="${item.quantity}"></td>
            <td><input type="number" value="${item.unit_price}"></td>
            <td><input type="number" value="${item.line_total}"></td>
        </tr>
    `).join('');
}

// Reload Image Toggle
function reloadImage() {
    if (!currentDocId) return;
    const isAnnotated = document.getElementById('toggle-annotated').checked;
    document.getElementById('doc-viewer-img').src = `/documents/${currentDocId}/image?annotated=${isAnnotated}`;
}

// Submit Human Review Corrections
async function submitReview(status) {
    if (!currentDocId) return;
    
    const payload = {
        approval_status: status,
        reviewer_id: "human_reviewer_01"
    };

    const res = await fetch(`/documents/${currentDocId}/review`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (res.ok) {
        alert(`Document ${status.toLowerCase()} successfully!`);
        showTab('history-tab');
    }
}

// Export Data Trigger
function exportData(format) {
    if (!currentDocId) return;
    window.open(`/documents/${currentDocId}/export?format=${format}`, '_blank');
}