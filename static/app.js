// DOM Elements
const eventForm = document.getElementById('eventForm');
const searchForm = document.getElementById('searchForm');
const searchSection = document.getElementById('searchSection');
const searchResults = document.getElementById('searchResults');
const paginationElement = document.getElementById('pagination');
const viewAllBtn = document.getElementById('viewAllBtn');
const hideAllBtn = document.getElementById('hideAllBtn');
const viewClustersBtn = document.getElementById('viewClustersBtn');
const step1Result = document.getElementById('step1Result');
const step2Result = document.getElementById('step2Result');
const searchInput = document.getElementById('searchInput');
const loadButton = document.getElementById('loadButton');
const searchButton = document.getElementById('searchButton');
const clusterSection = document.getElementById('clusterSection');
const clusterResults = document.getElementById('clusterResults');
const papersList = document.getElementById('papers-list');

// State Management
let currentPage = 1;
const resultsPerPage = 10;
let allPapers = [];
let filteredPapers = [];
let isViewingAll = false;
let currentClusters = null;

// Clear all displayed results
function clearResults(clearSearchInput = false) {
    searchResults.innerHTML = '';
    paginationElement.innerHTML = '';
    step2Result.textContent = '';
    step2Result.className = 'help';
    if (clearSearchInput) {
        searchInput.value = '';
        filteredPapers = [];
    }
}

// Show/hide sections
function showSection(section) {
    searchSection.style.display = 'none';
    clusterSection.style.display = 'none';
    searchResults.style.display = 'none';
    paginationElement.style.display = 'none';
    
    if (section === 'search') {
        searchSection.style.display = 'block';
        searchResults.style.display = 'block';
        paginationElement.style.display = 'block';
    } else if (section === 'clusters') {
        clusterSection.style.display = 'block';
    }
}

// Display cluster results
function displayClusterResults(clusters) {
    let html = '';
    clusters.forEach((cluster, index) => {
        html += `
            <div class="box cluster-box">
                <h3 class="title is-5">${cluster.task}</h3>
                <p class="subtitle is-6 mb-3">${cluster.description}</p>
                <div class="cluster-papers" id="cluster-${index}" style="display: none;">
                    <div class="papers-list mb-3"></div>
                    <div class="summary-section"></div>
                </div>
                <button class="button is-small is-info is-light view-cluster-btn" 
                        data-cluster-index="${index}" 
                        data-task-name="${cluster.task}"
                        data-paper-ids='${JSON.stringify(cluster.paper_ids)}'>
                    View Papers & Summary
                </button>
            </div>
        `;
    });
    clusterResults.innerHTML = html;

    // Add click handlers for cluster buttons
    document.querySelectorAll('.view-cluster-btn').forEach(btn => {
        btn.addEventListener('click', async function() {
            const clusterIndex = this.dataset.clusterIndex;
            const taskName = this.dataset.taskName;
            const paperIds = JSON.parse(this.dataset.paperIds);
            const clusterElement = document.getElementById(`cluster-${clusterIndex}`);
            const papersListElement = clusterElement.querySelector('.papers-list');
            const summaryElement = clusterElement.querySelector('.summary-section');
            
            // Toggle visibility
            if (clusterElement.style.display === 'none') {
                clusterElement.style.display = 'block';
                this.textContent = 'Hide Papers & Summary';
                
                // Display papers
                const clusterPapers = allPapers.filter(p => paperIds.includes(p.id));
                papersListElement.innerHTML = clusterPapers.map(paper => `
                    <div class="paper-card">
                        <h4>${paper.title}</h4>
                        <p class="abstract">${paper.abstract}</p>
                    </div>
                `).join('');
                
                // Fetch and display summary
                try {
                    const response = await fetch('/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            task_name: taskName,
                            paper_ids: paperIds
                        }),
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        summaryElement.innerHTML = `
                            <div class="box">
                                <h4 class="title is-6">Cluster Summary</h4>
                                <div class="content">${marked.parse(data.summary)}</div>
                            </div>
                        `;
                    } else {
                        summaryElement.innerHTML = `
                            <div class="notification is-danger">
                                Failed to load summary
                            </div>
                        `;
                    }
                } catch (error) {
                    summaryElement.innerHTML = `
                        <div class="notification is-danger">
                            Error: ${error.message}
                        </div>
                    `;
                }
            } else {
                clusterElement.style.display = 'none';
                this.textContent = 'View Papers & Summary';
            }
        });
    });
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

async function searchPapers(query) {
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: 50
            })
        });
        
        if (!response.ok) {
            throw new Error('Search failed');
        }
        
        const results = await response.json();
        return results;
    } catch (error) {
        console.error('Error searching papers:', error);
        throw error;
    }
}

function displayPaperResults(papers) {
    papersList.innerHTML = '';
    
    if (papers.length === 0) {
        papersList.innerHTML = '<div class="notification is-warning">No papers found matching your search.</div>';
        return;
    }
    
    papers.forEach(paper => {
        const paperElement = document.createElement('div');
        paperElement.className = 'box';
        paperElement.innerHTML = `
            <article class="media">
                <div class="media-content">
                    <div class="content">
                        <p>
                            <strong>${paper.title}</strong>
                            ${paper.score ? `<span class="tag is-info is-light ml-2">Score: ${paper.score.toFixed(3)}</span>` : ''}
                            <br>
                            <small>${paper.authors.join(', ')}</small>
                            <br>
                            ${paper.abstract}
                        </p>
                    </div>
                </div>
            </article>
        `;
        papersList.appendChild(paperElement);
    });
}

// Event Listeners
eventForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const eventUrl = document.getElementById('eventUrl').value;
    
    try {
        setLoadingPapers(true);
        const response = await fetch('/load_papers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ event_url: eventUrl })
        });
        
        if (!response.ok) {
            throw new Error('Failed to load papers');
        }
        
        allPapers = await response.json();
        
        // Show success message
        step1Result.textContent = `Successfully loaded ${allPapers.length} papers.`;
        step1Result.className = 'help is-success';
        
        // Show the search container and papers list
        document.getElementById('search-container').classList.remove('is-hidden');
        document.getElementById('papers-list').classList.remove('is-hidden');
        
        // Display all papers initially
        displayPaperResults(allPapers);
        
    } catch (error) {
        console.error('Error:', error);
        step1Result.textContent = 'Error loading papers: ' + error.message;
        step1Result.className = 'help is-danger';
    } finally {
        setLoadingPapers(false);
    }
});

// Handle search form submission
searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    if (!query) {
        // If search is empty, show all papers
        displayPaperResults(allPapers);
        return;
    }
    
    try {
        searchButton.classList.add('is-loading');
        const results = await searchPapers(query);
        displayPaperResults(results);
    } catch (error) {
        console.error('Error searching papers:', error);
        papersList.innerHTML = '<div class="notification is-danger">Error searching papers. Please try again.</div>';
    } finally {
        searchButton.classList.remove('is-loading');
    }
});

viewClustersBtn.addEventListener('click', startClustering);

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    try {
        setLoadingSearch(true);
        
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: 50
            })
        });

        const data = await response.json();
        
        if (data.error) {
            // Show error message
            searchResults.innerHTML = `
                <div class="notification is-warning">
                    ${data.error}
                </div>
            `;
            return;
        }

        // Update pagination info
        currentPage = 1;
        filteredPapers = data.results;
        displayResults(currentPage);
        displayPagination();
        step2Result.textContent = `Found ${filteredPapers.length} papers`;
        step2Result.className = 'help is-success';
        
    } catch (error) {
        console.error('Search error:', error);
        searchResults.innerHTML = `
            <div class="notification is-danger">
                Error performing search. Please try again later.
            </div>
        `;
    } finally {
        setLoadingSearch(false);
    }
}

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await performSearch();
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

searchInput.addEventListener('input', () => {
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

function showPapers(clusters) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    // Create container for cluster labels
    const clusterLabels = document.createElement('div');
    clusterLabels.className = 'cluster-labels';
    resultsDiv.appendChild(clusterLabels);

    // Create container for paper details
    const paperDetails = document.createElement('div');
    paperDetails.className = 'paper-details';
    paperDetails.style.display = 'none';
    resultsDiv.appendChild(paperDetails);

    // Sort clusters by size (descending), but keep "Uncategorized" at the end
    clusters.sort((a, b) => {
        if (a.task === "Uncategorized") return 1;
        if (b.task === "Uncategorized") return -1;
        return b.papers.length - a.papers.length;
    });

    clusters.forEach(cluster => {
        // Create cluster label button
        const labelBtn = document.createElement('button');
        labelBtn.className = 'cluster-label';
        labelBtn.textContent = `${cluster.task} (${cluster.papers.length})`;
        clusterLabels.appendChild(labelBtn);

        labelBtn.addEventListener('click', () => {
            // Hide any existing paper details
            paperDetails.style.display = 'block';
            paperDetails.innerHTML = '';

            // Create list of paper titles
            const titlesList = document.createElement('div');
            titlesList.className = 'paper-titles';
            
            cluster.papers.forEach(paper => {
                const titleBtn = document.createElement('button');
                titleBtn.className = 'paper-title';
                titleBtn.textContent = paper.title;
                titlesList.appendChild(titleBtn);

                // Create container for this paper's abstract
                const abstractDiv = document.createElement('div');
                abstractDiv.className = 'paper-abstract';
                abstractDiv.style.display = 'none';
                abstractDiv.innerHTML = `
                    <p><strong>Authors:</strong> ${paper.authors.join(', ')}</p>
                    <p><strong>Abstract:</strong> ${paper.abstract}</p>
                    <p><a href="${paper.link}" target="_blank" rel="noopener noreferrer">View Paper</a></p>
                `;
                titlesList.appendChild(abstractDiv);

                titleBtn.addEventListener('click', () => {
                    // Toggle abstract visibility
                    abstractDiv.style.display = abstractDiv.style.display === 'none' ? 'block' : 'none';
                });
            });

            paperDetails.appendChild(titlesList);
        });
    });
}

// Add CSS styles
const style = document.createElement('style');
style.textContent = `
.cluster-labels {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
}

.cluster-label {
    padding: 8px 16px;
    border: none;
    border-radius: 20px;
    background-color: #f0f0f0;
    cursor: pointer;
    transition: background-color 0.2s;
    font-size: 0.95em;
}

.cluster-label:hover {
    background-color: #e0e0e0;
}

.paper-titles {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 800px;
    margin: 0 auto;
}

.paper-title {
    text-align: left;
    padding: 12px;
    border: none;
    background: none;
    cursor: pointer;
    color: #2196F3;
    font-size: 1.1em;
    line-height: 1.4;
    transition: color 0.2s;
}

.paper-title:hover {
    color: #1976D2;
}

.paper-abstract {
    margin: 10px 20px;
    padding: 15px;
    background-color: #f8f8f8;
    border-radius: 8px;
    font-size: 0.95em;
    line-height: 1.6;
}

.paper-abstract p {
    margin: 10px 0;
}

.paper-abstract a {
    color: #2196F3;
    text-decoration: none;
}

.paper-abstract a:hover {
    text-decoration: underline;
}`;

document.head.appendChild(style);

async function startClustering() {
    try {
        // Start clustering in background
        const response = await fetch('/cluster');
        if (!response.ok) throw new Error('Failed to start clustering');
        
        // Start polling for status
        pollClusterStatus();
        
    } catch (error) {
        console.error('Error starting clustering:', error);
        showError('Failed to start clustering: ' + error.message);
    }
}

async function pollClusterStatus() {
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-indicator';
    loadingDiv.innerHTML = 'Clustering papers... <div class="spinner"></div>';
    resultsDiv.appendChild(loadingDiv);
    
    try {
        while (true) {
            const response = await fetch('/cluster_status');
            if (!response.ok) throw new Error('Failed to get clustering status');
            
            const data = await response.json();
            if (data.status === 'complete' && data.clusters) {
                loadingDiv.remove();
                showPapers(data.clusters.clusters);
                break;
            }
            
            // Wait before polling again
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    } catch (error) {
        console.error('Error polling cluster status:', error);
        loadingDiv.innerHTML = 'Error clustering papers: ' + error.message;
    }
}

// Add CSS for loading indicator
const loadingStyle = document.createElement('style');
loadingStyle.textContent = `
.loading-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin: 20px 0;
}

.spinner {
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}`;

document.head.appendChild(loadingStyle);
