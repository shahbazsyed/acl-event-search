// DOM Elements
const eventForm = document.getElementById('eventForm');
const searchForm = document.getElementById('searchForm');
const searchSection = document.getElementById('searchSection');
const searchResults = document.getElementById('searchResults');
const paginationElement = document.getElementById('pagination');
const viewAllBtn = document.getElementById('viewAllBtn');
const hideAllBtn = document.getElementById('hideAllBtn');
const step1Result = document.getElementById('step1Result');
const step2Result = document.getElementById('step2Result');
const searchQuery = document.getElementById('searchQuery');
const loadButton = document.getElementById('loadButton');
const searchButton = document.getElementById('searchButton');

// State
let currentPage = 1;
const resultsPerPage = 10;
let allPapers = [];
let filteredPapers = [];
let currentEventUrl = '';
let isViewingAll = false;

// Clear all displayed results
function clearResults(clearSearchInput = false) {
    searchResults.innerHTML = '';
    paginationElement.innerHTML = '';
    step2Result.textContent = '';
    step2Result.className = 'step-result';
    if (clearSearchInput) {
        searchQuery.value = '';
        filteredPapers = []; // Reset filtered papers when clearing search
    }
}

// Show/hide loading state for load papers button
function setLoadingPapers(isLoading) {
    loadButton.classList.toggle('is-loading', isLoading);
    if (isLoading) {
        loadButton.querySelector('span').textContent = 'Loading papers...';
    } else {
        loadButton.querySelector('span').textContent = 'Load Papers';
    }
}

// Show/hide loading state for search button
function setLoadingSearch(isLoading) {
    searchButton.classList.toggle('is-loading', isLoading);
    if (isLoading) {
        searchButton.querySelector('span').textContent = 'Searching...';
    } else {
        searchButton.querySelector('span').textContent = 'Search';
    }
}

// Event Listeners
eventForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const eventUrl = document.getElementById('eventUrl').value;

    try {
        setLoadingPapers(true);
        
        const response = await fetch(`/refresh/${encodeURIComponent(eventUrl)}`);
        const data = await response.json();
        
        if (response.ok) {
            setLoadingPapers(true);
            currentEventUrl = eventUrl;  // Store the current event URL
            step1Result.textContent = `Successfully loaded ${data.paper_count} papers`;
            step1Result.className = 'step-result success';
            searchSection.style.display = 'block';
            viewAllBtn.style.display = 'inline-block';
            
            // Get all papers for the view all functionality
            const papersResponse = await fetch(`/papers/${encodeURIComponent(eventUrl)}?skip=0&limit=1000`);
            allPapers = await papersResponse.json();
            clearResults(true); // Clear everything including search input when loading new papers
        } else {
            step1Result.textContent = `Error: ${data.detail}`;
        }
    } catch (error) {
        step1Result.textContent = `Error: ${error.message}`;
    } finally {
        setLoadingPapers(false);
    }
});

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('searchQuery').value;

    try {
        setLoadingSearch(true);
        
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                event_url: currentEventUrl,  // Include the event URL in the search
                top_k: 50
            }),
        });
        
        const results = await response.json();
        filteredPapers = results;
        step2Result.textContent = `Found ${filteredPapers.length} relevant papers`;
        step2Result.className = 'step-result success';
        displayResults(1);
    } catch (error) {
        searchResults.innerHTML = `Error: ${error.message}`;
    } finally {
        setLoadingSearch(false);
    }
});

viewAllBtn.addEventListener('click', () => {
    clearResults(true);
    displayAllPapers();
    viewAllBtn.style.display = 'none';
    hideAllBtn.style.display = 'inline-block';
    isViewingAll = true;
});

hideAllBtn.addEventListener('click', () => {
    clearResults(true);
    hideAllBtn.style.display = 'none';
    viewAllBtn.style.display = 'inline-block';
    isViewingAll = false;
});

searchQuery.addEventListener('input', () => {
    clearResults(false);
});

// Functions
function displayResults(page) {
    currentPage = page;
    const papers = isViewingAll ? allPapers : filteredPapers;
    
    const start = (page - 1) * resultsPerPage;
    const end = start + resultsPerPage;
    const paginatedPapers = papers.slice(start, end);

    searchResults.innerHTML = paginatedPapers.map(paper => `
        <div class="paper-card">
            <h4><a href="${paper.link}" target="_blank">${paper.title}</a></h4>
            ${paper.score ? `<div class="score">Score: ${paper.score}</div>` : ''}
            <div class="abstract">${paper.abstract || 'No abstract available'}</div>
        </div>
    `).join('');
    
    if (!isViewingAll) {
        displayPagination();
    }
}

function displayAllPapers() {
    searchResults.innerHTML = allPapers.map(paper => `
        <div class="paper-card">
            <h4><a href="${paper.link}" target="_blank">${paper.title}</a></h4>
            <div class="abstract">${paper.abstract || 'No abstract available'}</div>
        </div>
    `).join('');
    
    // Hide pagination when viewing all papers
    paginationElement.innerHTML = '';
}

function displayPagination() {
    const papers = isViewingAll ? allPapers : filteredPapers;
    const totalPages = Math.ceil(papers.length / resultsPerPage);
    
    if (totalPages <= 1 || isViewingAll) {
        paginationElement.innerHTML = '';
        return;
    }

    const prevDisabled = currentPage === 1 ? 'disabled' : '';
    const nextDisabled = currentPage === totalPages ? 'disabled' : '';

    paginationElement.innerHTML = `
        <i class="bi bi-chevron-left pagination-arrow" onclick="displayResults(${currentPage - 1})" ${prevDisabled}></i>
        <span>Page ${currentPage} of ${totalPages}</span>
        <i class="bi bi-chevron-right pagination-arrow" onclick="displayResults(${currentPage + 1})" ${nextDisabled}></i>
    `;
}
