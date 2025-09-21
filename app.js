
window.API_BASE_URL = window.API_BASE_URL || 'http://localhost:5026';

// Global variables
let isConnected = false;
// const API_BASE_URL = 'http://localhost:5026'; // replaced by window.API_BASE_URL
const FALLBACK_PORT = '5001';

// DOM elements
const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const resultSection = document.getElementById('resultSection');
const errorMsg = document.getElementById('errorMsg');
const statusText = document.getElementById('statusText');
const statusDot = document.querySelector('.status-dot');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Book Price Comparison App Initialized');
    
    // Test backend connection on page load
    testBackendConnection();
    
    // Set up event listeners
    setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
    searchForm.addEventListener('submit', handleSearch);
    searchInput.addEventListener('input', clearError);
}

// Handle search form submission
async function handleSearch(e) {
    e.preventDefault();
    
    const title = searchInput.value.trim();
    if (!title) {
        showError('Please enter a book title or ISBN to search.');
        return;
    }
    
    // Clear previous results and errors
    clearResults();
    clearError();
    
    // Show loading state
    setLoadingState(true);
    
    try {
        // Check backend connection first
        if (!isConnected) {
            await testBackendConnection();
            if (!isConnected) {
                throw new Error('Backend server is not accessible. Please make sure the Flask server is running.');
            }
        }
        
        // Search for the book
        const bookData = await searchBook(title);
        displayResults(bookData);
        
    } catch (error) {
        console.error('Search error:', error);
        showError(error.message);
    } finally {
        setLoadingState(false);
    }
}

// Test backend connectivity with fallback ports
async function testBackendConnection() {
    const ports = [Number(new URL(window.API_BASE_URL).port) || 5026, 5000, 5001, 8000, 3000];
    
    for (const port of ports) {
        try {
            updateStatus(`Testing connection on port ${port}...`, 'checking');
            
            const response = await fetch(`http://localhost:${port}/test`, {
                method: 'GET',
                timeout: 3000
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log(`✅ Backend connected on port ${port}:`, data.message);
                updateStatus(`Backend connected on port ${port}!`, 'connected');
                isConnected = true;
                
                // Update the API base URL to use the working port
                window.API_BASE_URL = `http://localhost:${port}`;
                return true;
            }
        } catch (error) {
            console.log(`❌ Port ${port} failed:`, error.message);
            continue;
        }
    }
    
    // If no ports work, show comprehensive error
    console.error('❌ Backend connection failed on all ports');
    updateStatus('Backend connection failed - server not running', 'error');
    isConnected = false;
    
    // Show helpful troubleshooting info
    showError(`
        <strong>Backend Connection Error</strong><br><br>
        The Flask server is not running or accessible.<br><br>
        <strong>To fix this:</strong><br>
        1. Open a new terminal/command prompt<br>
        2. Navigate to your project folder<br>
        3. Run: <code>python backend.py</code><br>
        4. Wait for "Running on http://0.0.0.0:5026" message<br>
        5. Refresh this page<br><br>
        <strong>Other tried ports:</strong> 5000, 5001, 8000, 3000
    `);
    
    return false;
}

// Search for a book
async function searchBook(title) {
    try {
        const baseUrl = window.API_BASE_URL;
        const response = await fetch(`${baseUrl}/search?title=${encodeURIComponent(title)}`, {
            method: 'GET',
            timeout: 10000
        });
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Book not found. Please try a different title or ISBN.');
        } else {
                throw new Error(`Server error: ${response.status} - ${response.statusText}`);
            }
        }
        
        const data = await response.json();
        return data;
        
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error. Please check your internet connection and ensure the backend server is running.');
        }
        throw error;
    }
}

function randomRatingOutOfFive() {
    const n = Math.floor(Math.random() * 4) + 1; // 1..4
    return `${n}/5`;
}

// Display search results
function displayResults(data) {
    // Fill Amazon details
    document.getElementById('amazonTitle').textContent = data.amazon_title || 'N/A';
    document.getElementById('amazonAuthor').textContent = data.amazon_author || 'N/A';
    document.getElementById('amazonISBN').textContent = data.amazon_isbn || 'N/A';
    document.getElementById('amazonPrice').textContent = formatPrice(data.amazon_price);
    // Random rating 1..4 out of 5
    document.getElementById('amazonRating').textContent = randomRatingOutOfFive();
    
    // Fill Flipkart details
    document.getElementById('flipkartTitle').textContent = data.flipkart_title || 'N/A';
    document.getElementById('flipkartAuthor').textContent = data.flipkart_author || 'N/A';
    document.getElementById('flipkartISBN').textContent = data.flipkart_isbn || 'N/A';
    document.getElementById('flipkartPrice').textContent = formatPrice(data.flipkart_price);
    // Random rating 1..4 out of 5
    document.getElementById('flipkartRating').textContent = randomRatingOutOfFive();
    
    // Fill summary information
    document.getElementById('cheapestPlatform').textContent = data.cheapest_platform || 'N/A';
    
    // Also show a random predicted rating 1..4 out of 5
    document.getElementById('predictedRating').textContent = randomRatingOutOfFive();
    
    // Highlight price differences
    highlightPrices();
    
    // Display recommendations
    displayRecommendations(data.recommendations || []);
    
    // Show results
    resultSection.style.display = 'block';
    
    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Helpers for clearing UI state
function clearResults() {
    try {
        resultSection.style.display = 'none';
        const rec = document.getElementById('recommendations');
        if (rec) rec.innerHTML = '';
        const fields = ['amazonTitle','amazonAuthor','amazonISBN','amazonPrice','amazonRating','flipkartTitle','flipkartAuthor','flipkartISBN','flipkartPrice','flipkartRating','cheapestPlatform','predictedRating'];
        fields.forEach(id => { const el = document.getElementById(id); if (el) el.textContent = ''; });
    } catch (e) { /* no-op */ }
}

function clearError() {
    try {
        errorMsg.style.display = 'none';
        errorMsg.innerHTML = '';
    } catch (e) { /* no-op */ }
}

function setLoadingState(loading) {
    if (loading) {
        if (searchBtn) { searchBtn.disabled = true; searchBtn.innerHTML = '<span>⏳ Searching...</span>'; }
        if (searchInput) { searchInput.disabled = true; }
    } else {
        if (searchBtn) { searchBtn.disabled = false; searchBtn.innerHTML = '<span>🔍 Search</span>'; }
        if (searchInput) { searchInput.disabled = false; }
    }
}

function updateStatus(message, status) {
    try {
        if (statusText) statusText.textContent = message;
        if (statusDot) statusDot.className = `status-dot ${status || ''}`;
    } catch (e) { /* no-op */ }
}

function showError(message) {
    try {
        if (!errorMsg) return;
        errorMsg.innerHTML = typeof message === 'string' ? message : (message && message.message) || 'An error occurred';
        errorMsg.style.display = 'block';
        // Auto-hide after 7s
        setTimeout(() => {
            if (errorMsg) errorMsg.style.display = 'none';
        }, 7000);
    } catch (e) { /* no-op */ }
}

function formatPrice(price) {
    if (price === null || price === undefined || price === 'N/A') return 'N/A';
    const n = parseFloat(price);
    if (isNaN(n)) return String(price);
    return `₹${n.toFixed(2)}`;
}

function formatRating(rating) {
    if (rating === null || rating === undefined || rating === 'N/A') return 'N/A';
    const n = parseFloat(rating);
    if (isNaN(n)) return String(rating);
    return `${n.toFixed(1)} ⭐`;
}

function highlightPrices() {
    const amazonPriceEl = document.getElementById('amazonPrice');
    const flipkartPriceEl = document.getElementById('flipkartPrice');
    if (!amazonPriceEl || !flipkartPriceEl) return;
    amazonPriceEl.classList.remove('lowest-price', 'highest-price');
    flipkartPriceEl.classList.remove('lowest-price', 'highest-price');
    const a = parseFloat((amazonPriceEl.textContent || '').replace('₹',''));
    const f = parseFloat((flipkartPriceEl.textContent || '').replace('₹',''));
    if (!isNaN(a) && !isNaN(f)) {
        if (a < f) {
            amazonPriceEl.classList.add('lowest-price');
            flipkartPriceEl.classList.add('highest-price');
        } else if (f < a) {
            flipkartPriceEl.classList.add('lowest-price');
            amazonPriceEl.classList.add('highest-price');
        } else {
            amazonPriceEl.classList.add('lowest-price');
            flipkartPriceEl.classList.add('lowest-price');
        }
    } else if (!isNaN(a)) {
        amazonPriceEl.classList.add('lowest-price');
    } else if (!isNaN(f)) {
        flipkartPriceEl.classList.add('lowest-price');
    }
}

function displayRecommendations(recs) {
    const recDiv = document.getElementById('recommendations');
    if (!recDiv) return;
    recDiv.innerHTML = '';
    if (!Array.isArray(recs) || recs.length === 0) {
        recDiv.innerHTML = '<p class="no-recommendations">No recommendations found.</p>';
        return;
    }
    recs.forEach(r => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.innerHTML = `<h3>${(r.title || 'Unknown')}</h3><p>by ${(r.author || 'Unknown')}</p>`;
        recDiv.appendChild(card);
    });
}

// Apply pink pill style on cheapest platform label after results fill
(function attachCheapestPillObserver(){
    const target = document.getElementById('cheapestPlatform');
    if (!target) return;
    target.classList.add('cheapest-pill');
})();

// Theme toggle
(function initThemeToggle(){
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const current = localStorage.getItem('theme') || 'light';
    if (current === 'dark') document.body.classList.add('dark');
    btn.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        const mode = document.body.classList.contains('dark') ? 'dark' : 'light';
        localStorage.setItem('theme', mode);
    });
})();
