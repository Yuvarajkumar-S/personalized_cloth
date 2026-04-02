// Show loading overlay on form submit
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recommendation-form');
    if (form) {
        form.addEventListener('submit', function() {
            const overlay = document.getElementById('loadingOverlay');
            if (overlay) overlay.classList.add('active');
        });
    }
});

// Reset form function
function resetForm() {
    const form = document.getElementById('recommendation-form');
    if (form) {
        form.reset();
        showToast('Form has been reset!', 'success');
    }
}

// Fill example function
function fillExample() {
    const hairSelect = document.getElementById('hair_color');
    const eyeSelect = document.getElementById('eye_color');
    const skinSelect = document.getElementById('skin_tone');
    const underSelect = document.getElementById('under_tone');
    const torsoSelect = document.getElementById('torso_length');
    const bodySelect = document.getElementById('body_proportion');
    
    if (hairSelect) hairSelect.value = 'Black';
    if (eyeSelect) eyeSelect.value = 'Brown';
    if (skinSelect) skinSelect.value = 'Medium';
    if (underSelect) underSelect.value = 'Warm';
    if (torsoSelect) torsoSelect.value = 'Balanced';
    if (bodySelect) bodySelect.value = 'Hourglass';
    
    showToast('Example profile loaded! Click "Get Recommendations"', 'success');
}

// Share website function
function shareWebsite() {
    if (navigator.share) {
        navigator.share({
            title: 'StyleAI - Personal Clothing Recommender',
            text: 'Find your perfect style with AI-powered recommendations!',
            url: window.location.href
        }).catch(() => copyToClipboard());
    } else {
        copyToClipboard();
    }
}

// Copy to clipboard
function copyToClipboard() {
    navigator.clipboard.writeText(window.location.href).then(() => {
        showToast('Link copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy link', 'error');
    });
}

// Print recommendations
function printRecommendations() {
    window.print();
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `${type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ'} ${message}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// Form validation
function validateForm() {
    const selects = document.querySelectorAll('#recommendation-form select');
    for (let select of selects) {
        if (!select.value) {
            showToast('Please fill in all fields', 'error');
            return false;
        }
    }
    return true;
}

// Hide loading overlay on page load
window.addEventListener('load', function() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('active');
});