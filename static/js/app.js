// === Global State ===
const state = {
    sessionId: null,
    taskId: null,
    filePath: null,
    isUploading: false,
    isPaused: false,
    uploadController: null,
    processingInterval: null,
    segments: [],
    translations: {},
    currentTab: 'original',
    videoPlayer: null,
    activeSegment: -1,
    displayLanguage: 'original',
    videoUrl: null,
    isVideo: false,
    pixelsPerSecond: 50,
    zoomLevel: 1.0,
    selectedSegment: null,
    isDragging: false,
    isDraggingPlayhead: false,
    subtitlePosition: 'bottom',
    isPanning: false,
    panStartX: 0,
    panStartScroll: 0,
    dragTarget: null,
    dragStartX: 0,
    dragStartOffset: 0,
    resizeType: null, // 'start' or 'end'
    exportLanguage: 'original'
};

// === DOM Elements ===
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadProgress: document.getElementById('uploadProgress'),
    uploadFilename: document.getElementById('uploadFilename'),
    uploadPercentage: document.getElementById('uploadPercentage'),
    uploadBar: document.getElementById('uploadBar'),
    chunkIndicators: document.getElementById('chunkIndicators'),
    uploadSpeed: document.getElementById('uploadSpeed'),
    uploadETA: document.getElementById('uploadETA'),
    filePreview: document.getElementById('filePreview'),
    videoPreview: document.getElementById('videoPreview'),
    imagePreview: document.getElementById('imagePreview'),
    previewInfo: document.getElementById('previewInfo'),
    startButton: document.getElementById('startButton'),
    processingSection: document.getElementById('processingSection'),
    processingMessage: document.getElementById('processingMessage'),
    processingBar: document.getElementById('processingBar'),
    processingPercentage: document.getElementById('processingPercentage'),
    playerSection: document.getElementById('playerSection'),
    mainVideoPlayer: document.getElementById('mainVideoPlayer'),
    subtitleOverlay: document.getElementById('subtitleOverlay'),
    timeDisplay: document.getElementById('timeDisplay'),
    resultsSection: document.getElementById('resultsSection'),
    segmentsList: document.getElementById('segmentsList'),
    fullTextView: document.getElementById('fullTextView'),
    fullTextEditor: document.getElementById('fullTextEditor'),
    translationResults: document.getElementById('translationResults'),
    translationsContainer: document.getElementById('translationsContainer'),
    translationTab: document.getElementById('translationTab'),
    toastContainer: document.getElementById('toastContainer'),
    deviceBadge: document.getElementById('deviceBadge'),
    deviceText: document.getElementById('deviceText'),
    timelineSection: document.getElementById('timelineSection'),
    timelineContainer: document.getElementById('timelineContainer'),
    timelineContent: document.getElementById('timelineContent'),
    timelineSegments: document.getElementById('timelineSegments'),
    timelinePlayhead: document.getElementById('timelinePlayhead'),
    timelineRuler: document.getElementById('timelineRuler'),
    zoomLevel: document.getElementById('zoomLevel'),
    subtitleLangSelect: document.getElementById('subtitleLangSelect')
};

// === Initialization ===
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initLanguages();
    initModels();
    initVideoPlayer();
    checkDevice();
    toggleEngineOptions(); // Initializa engine specific UI
});

function initUpload() {
    // Drag & Drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('drag-over');
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('drag-over');
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });
}

async function initLanguages() {
    try {
        const response = await fetch('/api/languages');
        const languages = await response.json();
        
        const languageSelect = document.getElementById('languageSelect');
        const targetLanguageSelect = document.getElementById('targetLanguageSelect');
        
        // Păstrează opțiunea auto
        languageSelect.innerHTML = '<option value="auto" selected>🔍 Detectare automată</option>';
        targetLanguageSelect.innerHTML = '';
        
        Object.entries(languages).forEach(([code, name]) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            if (code !== 'auto') {
                languageSelect.appendChild(option.cloneNode(true));
            }
            targetLanguageSelect.appendChild(option);
        });
        
        // Selectează Română implicit pentru limba țintă
        const roOption = targetLanguageSelect.querySelector('option[value="ro"]');
        if (roOption) roOption.selected = true;
        
    } catch (error) {
        console.error('Error loading languages:', error);
    }
}

async function initModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        document.getElementById('deviceText').textContent = data.device;
        
        if (data.device === 'cuda') {
            elements.deviceBadge.querySelector('.status-dot').style.background = 'var(--accent)';
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function initVideoPlayer() {
    const video = document.getElementById('mainVideoPlayer');
    if (!video) return;

    state.videoPlayer = video;
    elements.mainVideoPlayer = video;
    
    video.addEventListener('timeupdate', () => {
        if (state.segments.length > 0) {
            updateSubtitleDisplay(video.currentTime);
            updateActiveSegment(video.currentTime);
            updateTimelinePlayhead(video.currentTime);
        }
        updateTimeDisplay();
    });
    
    video.addEventListener('loadedmetadata', () => {
        updateTimeDisplay();
        if (state.segments.length > 0) {
            renderTimeline();
        }
    });
    
    video.addEventListener('play', () => {
        console.log('Video playing');
    });
    
    video.addEventListener('pause', () => {
        console.log('Video paused');
    });
    
    video.addEventListener('error', (e) => {
        console.error('Video error:', e);
        showToast('Eroare la încărcarea video-ului', 'error');
    });
    
    video.addEventListener('loadeddata', () => {
        console.log('Video loaded, duration:', video.duration);
        updateTimeDisplay();
    });
}

async function checkDevice() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        elements.deviceText.textContent = data.device.toUpperCase();
    } catch (error) {
        console.error('Device check error:', error);
    }
}

// === File Handling ===
async function handleFile(file) {
    // Validate file size
    if (file.size > 50 * 1024 * 1024 * 1024) {
        showToast('Fișierul depășește limita de 50GB', 'error');
        return;
    }
    
    // Verifică dacă e video sau audio
    state.isVideo = file.type.startsWith('video/');
    const fileExt = file.name.split('.').pop().toLowerCase();
    const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'mxf'];
    if (videoExts.includes(fileExt)) {
        state.isVideo = true;
    }
    
    // Show preview in unified player
    const url = URL.createObjectURL(file);
    state.videoUrl = url;
    elements.playerSection.style.display = 'block';
    elements.mainVideoPlayer.src = url;
    elements.mainVideoPlayer.load();
    
    // Start upload
    await startUpload(file);
}

async function startUpload(file) {
    let chunkSize = 10 * 1024 * 1024; // Default 10MB
    let totalChunks = Math.ceil(file.size / chunkSize);
    
    try {
        // Initialize upload session
        const initResponse = await fetch('/api/upload/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: file.name,
                total_size: file.size,
                total_chunks: totalChunks
            })
        });
        
        if (!initResponse.ok) {
            const errorData = await initResponse.json();
            throw new Error(errorData.error || 'Failed to initialize upload');
        }

        const initData = await initResponse.json();
        state.sessionId = initData.session_id;
        if (initData.chunk_size) chunkSize = initData.chunk_size;
        totalChunks = Math.ceil(file.size / chunkSize); // Re-calculate just in case

        state.isUploading = true;
        
        // Show progress
        elements.uploadArea.style.display = 'none';
        elements.uploadProgress.style.display = 'block';
        elements.uploadFilename.textContent = file.name;
        
        // Create chunk indicators
        createChunkIndicators(totalChunks);
        
        // Upload chunks
        const startTime = Date.now();
        let uploadedBytes = 0;
        
        for (let i = 0; i < totalChunks; i++) {
            if (!state.isUploading) break;
            
            while (state.isPaused) {
                await sleep(100);
                if (!state.isUploading) break;
            }
            
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const chunk = file.slice(start, end);
            
            const formData = new FormData();
            formData.append('session_id', state.sessionId);
            formData.append('chunk_number', i);
            formData.append('chunk', chunk);
            
            const response = await fetch('/api/upload/chunk', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Failed to upload chunk ${i}`);
            }

            const data = await response.json();
            
            // Update progress
            uploadedBytes += chunk.size;
            const progress = (uploadedBytes / file.size) * 100;
            updateUploadProgress(progress, i, totalChunks, uploadedBytes, startTime, file.size);
        }
        
        if (!state.isUploading) return;
        
        // Complete upload
        const completeResponse = await fetch('/api/upload/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                total_chunks: totalChunks
            })
        });
        
        if (!completeResponse.ok) {
            const errorData = await completeResponse.json();
            throw new Error(errorData.error || 'Failed to complete upload');
        }

        const completeData = await completeResponse.json();
        state.taskId = completeData.task_id;
        state.filePath = completeData.file_path;
        
        
        showToast('Upload complet! Puteți începe procesarea.', 'success');
        elements.startButton.style.display = 'flex';
        
    } catch (error) {
        console.error('Upload error:', error);
        showToast('Eroare la upload: ' + error.message, 'error');
        resetUpload();
    }
}

function createChunkIndicators(totalChunks) {
    elements.chunkIndicators.innerHTML = '';
    const maxDisplay = Math.min(totalChunks, 100);
    const step = Math.max(1, Math.ceil(totalChunks / maxDisplay));
    
    for (let i = 0; i < totalChunks; i += step) {
        const dot = document.createElement('div');
        dot.className = 'chunk-dot';
        dot.dataset.chunk = i;
        elements.chunkIndicators.appendChild(dot);
    }
}

function updateUploadProgress(progress, chunkIndex, totalChunks, uploadedBytes, startTime, fileSize) {
    elements.uploadPercentage.textContent = Math.round(progress) + '%';
    elements.uploadBar.style.width = progress + '%';
    
    // Update chunk indicators
    const dots = elements.chunkIndicators.children;
    if (dots.length > 0) {
        const dotIndex = Math.floor((chunkIndex / totalChunks) * dots.length);
        for (let i = 0; i <= dotIndex && i < dots.length; i++) {
            dots[i].classList.add('uploaded');
        }
    }
    
    // Calculate speed
    const elapsed = (Date.now() - startTime) / 1000;
    if (elapsed > 0) {
        const speed = uploadedBytes / elapsed;
        elements.uploadSpeed.textContent = formatSpeed(speed);
        
        // Calculate ETA
        const remainingBytes = fileSize - uploadedBytes;
        const eta = remainingBytes / speed;
        elements.uploadETA.textContent = formatTime(eta);
    }
}

function pauseUpload() {
    state.isPaused = !state.isPaused;
    const btn = document.getElementById('pauseUpload');
    btn.textContent = state.isPaused ? '▶️ Continuă' : '⏸️ Pauză';
}

function cancelUpload() {
    state.isUploading = false;
    state.isPaused = false;
    resetUpload();
    showToast('Upload anulat', 'warning');
}

function resetUpload() {
    state.isUploading = false;
    state.isPaused = false;
    elements.uploadArea.style.display = 'block';
    elements.uploadProgress.style.display = 'none';
    elements.uploadBar.style.width = '0%';
    elements.uploadPercentage.textContent = '0%';
}

// === Processing ===
function toggleEngineOptions() {
    const engine = document.getElementById('engineSelect').value;
    const whisperGroup = document.getElementById('whisperModelGroup');
    const coherePromptGroup = document.getElementById('coherePromptGroup');

    if (engine === 'whisper') {
        whisperGroup.style.display = 'block';
        if (coherePromptGroup) coherePromptGroup.style.display = 'none';
    } else {
        whisperGroup.style.display = 'none';
        if (coherePromptGroup) coherePromptGroup.style.display = 'block';
    }
}

async function startProcessing() {
    if (!state.taskId) {
        showToast('Încărcați mai întâi un fișier', 'warning');
        return;
    }
    
    // Check audio only mode
    const audioOnly = document.getElementById('audioOnly').checked;
    
    // Collect options
    const engine = document.getElementById('engineSelect').value;
    const model = document.getElementById('modelSelect').value;

    console.log('Starting processing with engine:', engine, 'model:', model);

    const options = {
        engine: engine,
        model: model,
        use_custom_prompt: document.getElementById('useCoherePrompt')?.checked,
        custom_prompt: document.getElementById('coherePrompt')?.value,
        language: document.getElementById('languageSelect').value,
        min_duration: parseFloat(document.getElementById('minDuration').value),
        max_duration: parseFloat(document.getElementById('maxDuration').value),
        max_chars: parseInt(document.getElementById('maxChars').value),
        use_vad: document.getElementById('useVAD').checked,
        use_margin: document.getElementById('useMargin').checked,
        isolate_voice: document.getElementById('isolateVoice').checked,
        deduplicate: document.getElementById('deduplicate').checked,
        prevent_overlap: document.getElementById('preventOverlap').checked,
        audio_only: audioOnly
    };
    
    if (!audioOnly) {
        options.translate = document.getElementById('enableTranslation').checked;
        if (options.translate) {
            options.target_languages = [document.getElementById('targetLanguageSelect').value];
            options.translation_engine = document.getElementById('translationEngine').value;
            options.llm_model = document.getElementById('llmModelSelect').value;
            options.custom_prompt = document.getElementById('customPrompt').value;
            
            // Adaugă limbile suplimentare
            const additionalSelects = document.querySelectorAll('.translation-lang-select');
            additionalSelects.forEach(select => {
                options.target_languages.push(select.value);
            });
        }
    }
    
    try {
        const response = await fetch('/api/process/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: state.taskId,
                file_path: state.filePath,
                options: options
            })
        });
        
        const data = await response.json();
        
        // Show processing section
        elements.startButton.style.display = 'none';
        elements.processingSection.style.display = 'block';
        elements.processingMessage.textContent = 'Se inițializează procesarea...';
        elements.processingBar.style.width = '0%';
        elements.processingPercentage.textContent = '0%';
        
        // Start polling
        startPolling(state.taskId);
        
    } catch (error) {
        console.error('Processing error:', error);
        showToast('Eroare la pornirea procesării', 'error');
    }
}

function startPolling(taskId) {
    if (state.processingInterval) {
        clearInterval(state.processingInterval);
    }
    
    state.processingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/process/status/${taskId}`);
            const data = await response.json();
            
            updateProcessingStatus(data);
            
            if (data.status === 'completed') {
                clearInterval(state.processingInterval);
                state.processingInterval = null;
                // Ia rezultatele
                await fetchResults(taskId);
                showToast('Procesare completă!', 'success');
            } else if (data.status === 'failed') {
                clearInterval(state.processingInterval);
                state.processingInterval = null;
                showToast('Eroare: ' + (data.error || 'Eroare necunoscută'), 'error');
                elements.processingSection.style.display = 'none';
                elements.startButton.style.display = 'flex';
            } else if (data.status === 'cancelled') {
                clearInterval(state.processingInterval);
                state.processingInterval = null;
                showToast('Procesare anulată', 'warning');
                elements.processingSection.style.display = 'none';
                elements.startButton.style.display = 'flex';
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000); // Polling la fiecare 2 secunde
}

async function fetchResults(taskId) {
    try {
        const response = await fetch(`/api/process/result/${taskId}`);
        const result = await response.json();
        showResults(result);
    } catch (error) {
        console.error('Error fetching results:', error);
        showToast('Eroare la obținerea rezultatelor', 'error');
    }
}

function updateProcessingStatus(data) {
    const { progress, message, status } = data;
    
    elements.processingMessage.textContent = message || status;
    elements.processingBar.style.width = (progress || 0) + '%';
    elements.processingPercentage.textContent = Math.round(progress || 0) + '%';
}

function cancelProcessing() {
    if (state.taskId) {
        fetch(`/api/process/cancel/${state.taskId}`, { method: 'POST' });
        if (state.processingInterval) {
            clearInterval(state.processingInterval);
            state.processingInterval = null;
        }
        elements.processingSection.style.display = 'none';
        elements.startButton.style.display = 'flex';
    }
}

// === Results Display ===
function showResults(result) {
    state.segments = result.segments || [];
    state.translations = result.translations || {};
    state.displayLanguage = 'original';
    
    console.log('Showing results:', state.segments.length, 'segments');
    console.log('Is video:', state.isVideo);
    console.log('Task ID:', state.taskId);
    
    elements.processingSection.style.display = 'none';
    elements.resultsSection.style.display = 'block';
    elements.playerSection.style.display = 'block';
    
    // Use the server-side file once processed
    const mediaUrl = state.isVideo ? `/api/video/${state.taskId}` : `/api/audio/${state.taskId}`;
    console.log('Setting media URL:', mediaUrl);

    // Only reload if the URL is different to avoid interruption
    if (!elements.mainVideoPlayer.src.endsWith(mediaUrl)) {
        elements.mainVideoPlayer.src = mediaUrl;
        elements.mainVideoPlayer.load();
    }
    
    // Show translations if available
    if (Object.keys(state.translations).length > 0) {
        elements.translationResults.style.display = 'block';
        elements.translationTab.style.display = 'inline-block';
        displayTranslations();
        updateSubtitleLangSelect();
    } else {
        elements.translationResults.style.display = 'none';
        elements.translationTab.style.display = 'none';
        if (elements.subtitleLangSelect) elements.subtitleLangSelect.style.display = 'none';
    }
    
    // Render segments
    renderSegments();
    updateFullText();
    
    // Scroll to top of segments
    elements.segmentsList.scrollTop = 0;

    // Show and render timeline
    elements.timelineSection.style.display = 'block';
    renderTimeline();

    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function renderSegments() {
    elements.segmentsList.innerHTML = '';
    
    if (state.segments.length === 0) {
        elements.segmentsList.innerHTML = '<div class="no-segments">Nu există segmente</div>';
        return;
    }
    
    state.segments.forEach((segment, index) => {
        const div = document.createElement('div');
        div.className = 'segment-item';
        div.dataset.index = index;
        div.dataset.start = segment.start;
        div.dataset.end = segment.end;
        
        let displayText = segment.text;
        if (state.displayLanguage !== 'original' && state.translations[state.displayLanguage]) {
            displayText = state.translations[state.displayLanguage][index];
        }

        if (state.selectedSegment === index) {
            div.classList.add('selected');
        }

        div.innerHTML = `
            <div class="segment-number">${index + 1}</div>
            <div class="segment-content">
                <div class="segment-text" contenteditable="true" 
                     onblur="updateSegment(${index}, this.textContent)">
                    ${escapeHtml(displayText || '')}
                </div>
                <div class="segment-time">
                    ${formatTimestamp(segment.start)} → ${formatTimestamp(segment.end)}
                </div>
            </div>
            <div class="segment-actions">
                <button class="segment-play-btn" onclick="event.stopPropagation(); seekToTime(${segment.start})" title="Redă de aici">
                    ▶️
                </button>
                <button class="segment-edit-btn" onclick="event.stopPropagation(); editSegment(${index})" title="Editează">
                    ✏️
                </button>
                <button class="segment-delete-btn" onclick="event.stopPropagation(); deleteSegment(${index})" title="Șterge">
                    🗑️
                </button>
            </div>
        `;
        
        div.addEventListener('click', (e) => {
            const isButton = e.target.closest('button');
            const isText = e.target.classList.contains('segment-text');

            if (isButton) return; // Let button handlers (edit/delete/play) handle it

            if (state.selectedSegment === index) {
                if (isText) return; // Already editing or focused on text
            }

            state.selectedSegment = index;

            if (!isText) {
                seekToTime(segment.start, false);
            }

            renderSegments();
            renderTimeline();

            // If we clicked on text but it wasn't selected yet, we need to re-focus after render
            if (isText) {
                setTimeout(() => editSegment(index), 0);
            }
        });
        
        elements.segmentsList.appendChild(div);
    });
}

function updateSegment(index, text) {
    if (state.segments[index]) {
        if (state.displayLanguage === 'original') {
            state.segments[index].text = text.trim();
        } else if (state.translations[state.displayLanguage]) {
            state.translations[state.displayLanguage][index] = text.trim();
        }

        updateFullText();
        displayTranslations(); // Sync the other panel
        renderTimeline(); // Sync timeline labels

        console.log('Segment updated:', index, 'Lang:', state.displayLanguage);
    }
}

function editSegment(index) {
    const segmentElement = document.querySelector(`.segment-item[data-index="${index}"]`);
    const textElement = segmentElement.querySelector('.segment-text');
    textElement.focus();
    
    // Selectează tot textul
    const range = document.createRange();
    range.selectNodeContents(textElement);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
}

function updateFullText() {
    let texts = [];
    if (state.currentTab === 'original') {
        texts = state.segments.map(s => s.text || '');
    } else {
        // Translation tab
        if (state.displayLanguage === 'original') {
            // If display language is original but on translation tab, show first available translation
            const availableLangs = Object.keys(state.translations);
            if (availableLangs.length > 0) {
                texts = state.translations[availableLangs[0]];
            } else {
                texts = state.segments.map(s => s.text || '');
            }
        } else {
            texts = state.translations[state.displayLanguage] || state.segments.map(s => s.text || '');
        }
    }

    elements.fullTextEditor.value = texts.join('\n\n');
}

function displayTranslations() {
    elements.translationsContainer.innerHTML = '';
    
    Object.entries(state.translations).forEach(([lang, texts]) => {
        const langName = getLanguageName(lang);
        const div = document.createElement('div');
        div.className = 'translation-group glass-container';
        div.style.marginBottom = '20px';
        div.style.padding = '15px';
        
        let translationsHtml = '<div class="translation-segments">';
        texts.forEach((text, i) => {
            translationsHtml += `
                <div class="translation-segment" style="margin-bottom: 8px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                    <span style="color: var(--text-muted); font-size: 0.8rem;">${i + 1}.</span>
                    <span contenteditable="true"
                          onblur="updateTranslationSegment('${lang}', ${i}, this.textContent)"
                          style="margin-left: 8px; display: inline-block; width: calc(100% - 30px); outline: none;">
                        ${escapeHtml(text || '')}
                    </span>
                </div>
            `;
        });
        translationsHtml += '</div>';
        
        div.innerHTML = `
            <div class="translation-group-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: var(--primary); margin: 0;">🌐 ${langName}</h3>
                <div class="group-actions">
                    <button class="btn btn-sm" onclick="exportSRT('${lang}')">💾 SRT</button>
                    <button class="btn btn-sm" onclick="showDOCXDialog('${lang}')">📄 DOCX</button>
                </div>
            </div>
            ${translationsHtml}
        `;
        
        elements.translationsContainer.appendChild(div);
    });
}

function updateTranslationSegment(lang, index, text) {
    if (state.translations[lang] && state.translations[lang][index] !== undefined) {
        const oldText = state.translations[lang][index];
        const newText = text.trim();

        if (oldText !== newText) {
            state.translations[lang][index] = newText;

            // Only update downstream components that don't cause a re-render of this list
            if (state.displayLanguage === lang) {
                updateFullText();
                updateSubtitleDisplay(elements.mainVideoPlayer.currentTime);
                renderTimeline(); // Update labels on timeline
            }

            console.log(`Translation updated for ${lang} at ${index}`);
        }
    }
}

// === Video Player Controls ===
function seekToTime(time, autoPlay = true) {
    console.log('Seeking to:', time, 'autoPlay:', autoPlay);
    if (elements.mainVideoPlayer) {
        elements.mainVideoPlayer.currentTime = time;
        if (autoPlay && elements.mainVideoPlayer.paused) {
            elements.mainVideoPlayer.play().catch(e => console.log('Play error:', e));
        }
    }
}

function highlightSegment(index) {
    // Remove previous highlight
    const prevActive = document.querySelectorAll('.segment-item.active');
    prevActive.forEach(el => el.classList.remove('active'));
    
    // Add new highlight
    const newActive = document.querySelector(`.segment-item[data-index="${index}"]`);
    if (newActive) {
        newActive.classList.add('active');
    }
    
    state.activeSegment = index;
}

function togglePlayPause() {
    const video = elements.mainVideoPlayer;
    if (video.paused) {
        video.play().catch(e => console.log('Play error:', e));
    } else {
        video.pause();
    }
}

function updateSubtitleDisplay(currentTime) {
    const activeIndices = [];
    state.segments.forEach((s, i) => {
        if (currentTime >= s.start && currentTime <= s.end) {
            activeIndices.push(i);
        }
    });

    if (activeIndices.length > 0) {
        const html = activeIndices.map(index => {
            let text = '';
            if (state.displayLanguage === 'original') {
                text = state.segments[index].text;
            } else if (state.translations[state.displayLanguage]) {
                text = state.translations[state.displayLanguage][index];
            }

            const lines = text.split('\n');
            return lines.map(line => escapeHtml(line)).join('<br>');
        }).join('<br><hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.3); margin: 4px 0;"><br>');

        elements.subtitleOverlay.innerHTML = html;
        elements.subtitleOverlay.style.display = 'block';
    } else {
        elements.subtitleOverlay.textContent = '';
        elements.subtitleOverlay.style.display = 'none';
    }
}

function updateActiveSegment(currentTime) {
    const activeIndices = [];
    state.segments.forEach((s, i) => {
        if (currentTime >= s.start && currentTime <= s.end) {
            activeIndices.push(i);
        }
    });
    
    // Highlight first active for scrolling list
    if (activeIndices.length > 0) {
        const firstIndex = activeIndices[0];
        if (firstIndex !== state.activeSegment) {
            highlightSegment(firstIndex);
        }
    }

    // Highlight all on timeline
    document.querySelectorAll('.timeline-segment-block').forEach((el, i) => {
        if (activeIndices.includes(i)) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

function findSegmentAtTime(time) {
    return state.segments.find(s => time >= s.start && time <= s.end) || null;
}

function updateTimeDisplay() {
    const video = elements.mainVideoPlayer;
    if (video && !isNaN(video.currentTime) && !isNaN(video.duration)) {
        const current = formatTimestamp(video.currentTime);
        const duration = formatTimestamp(video.duration);
        elements.timeDisplay.textContent = `${current} / ${duration}`;
    } else {
        elements.timeDisplay.textContent = '00:00:00,000 / 00:00:00,000';
    }
}

// Adăugăm și funcționalitate pentru bara de progres
elements.mainVideoPlayer.addEventListener('seeking', () => {
    updateTimeDisplay();
});

elements.mainVideoPlayer.addEventListener('seeked', () => {
    updateTimeDisplay();
    updateSubtitleDisplay(elements.mainVideoPlayer.currentTime);
    updateActiveSegment(elements.mainVideoPlayer.currentTime);
});

// === Export Functions ===
async function exportSRT(lang = null) {
    if (!lang) lang = state.displayLanguage;

    if (state.segments.length === 0) {
        showToast('Nu există segmente de exportat', 'warning');
        return;
    }
    
    const segmentsToExport = state.segments.map((seg, i) => {
        let text = seg.text;
        if (lang !== 'original' && state.translations[lang]) {
            text = state.translations[lang][i];
        }
        return { ...seg, text: text };
    });

    try {
        const response = await fetch('/api/export/srt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                segments: segmentsToExport,
                legacy_diacritics: document.getElementById('docxLegacyDiacritics')?.checked || false
            })
        });
        
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const filename = lang === 'original' ? 'subtitles_original.srt' : `subtitles_${lang}.srt`;
        downloadFile(blob, filename);
        showToast(`SRT (${lang}) exportat cu succes!`, 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast('Eroare la export SRT', 'error');
    }
}

async function exportDOCX() {
    if (state.segments.length === 0) {
        showToast('Nu există segmente de exportat', 'warning');
        return;
    }
    
    const lang = state.exportLanguage || state.displayLanguage;
    const segmentsToExport = state.segments.map((seg, i) => {
        let text = seg.text;
        if (lang !== 'original' && state.translations[lang]) {
            text = state.translations[lang][i];
        }
        return { ...seg, text: text };
    });

    const metadata = {
        title: document.getElementById('docxTitle').value || '',
        series: document.getElementById('docxSeries').value || '',
        translator: document.getElementById('docxTranslator').value || '',
        editor: document.getElementById('docxEditor').value || ''
    };
    
    try {
        const response = await fetch('/api/export/docx', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                segments: segmentsToExport,
                metadata: metadata,
                legacy_diacritics: document.getElementById('docxLegacyDiacritics')?.checked || false
            })
        });
        
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const filename = lang === 'original' ? 'translation_original.docx' : `translation_${lang}.docx`;
        downloadFile(blob, filename);
        closeDOCXDialog();
        showToast(`DOCX (${lang}) exportat cu succes!`, 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast('Eroare la export DOCX', 'error');
    }
}

function copyFullText() {
    const text = elements.fullTextEditor.value;
    if (!text) {
        showToast('Nu există text de copiat', 'warning');
        return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
        showToast('Text copiat în clipboard!', 'success');
    }).catch(() => {
        // Fallback pentru browsere care nu suportă clipboard API
        elements.fullTextEditor.select();
        document.execCommand('copy');
        showToast('Text copiat în clipboard!', 'success');
    });
}

function showDOCXDialog(lang = null) {
    state.exportLanguage = lang || state.displayLanguage;
    document.getElementById('docxModal').style.display = 'flex';
}

function closeDOCXDialog() {
    document.getElementById('docxModal').style.display = 'none';
}

// === Translation Functions ===
function toggleTranslation() {
    const enabled = document.getElementById('enableTranslation').checked;
    document.getElementById('translationSettings').style.display = enabled ? 'block' : 'none';
    
    if (enabled) {
        updateModelOptions();
    }
}

function updateModelOptions() {
    const engine = document.getElementById('translationEngine').value;
    const promptGroup = document.getElementById('promptGroup');
    const llmModelGroup = document.getElementById('llmModelGroup');

    if (engine === 'llm' || engine === 'vllm') {
        promptGroup.style.display = 'block';
        llmModelGroup.style.display = 'block';
    } else {
        promptGroup.style.display = 'none';
        llmModelGroup.style.display = 'none';
    }
}

function addTranslationLanguage() {
    const container = document.getElementById('additionalLanguages');
    const div = document.createElement('div');
    div.className = 'setting-row';
    div.style.marginTop = '8px';
    
    const targetSelect = document.getElementById('targetLanguageSelect');
    div.innerHTML = `
        <select class="setting-select translation-lang-select">
            ${targetSelect.innerHTML}
        </select>
        <button class="btn btn-sm btn-cancel" onclick="this.parentElement.remove()">❌</button>
    `;
    container.appendChild(div);
}

// === Tabs ===
function switchTab(tab) {
    state.currentTab = tab;
    
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    if (event && event.target) {
        event.target.classList.add('active');
    }
    
    if (tab === 'original') {
        state.displayLanguage = 'original';
        elements.fullTextView.style.display = 'none';
        document.getElementById('segmentsContainer').style.display = 'block';
    } else {
        // Switch to the first translation if available
        const availableLangs = Object.keys(state.translations);
        if (availableLangs.length > 0 && state.displayLanguage === 'original') {
            state.displayLanguage = availableLangs[0];
        }

        document.getElementById('segmentsContainer').style.display = 'none';
        elements.fullTextView.style.display = 'block';
        updateFullText();
    }

    if (elements.subtitleLangSelect) {
        elements.subtitleLangSelect.value = state.displayLanguage;
    }

    renderSegments();
    renderTimeline();
}

function toggleSubtitlePosition() {
    const isTop = document.getElementById('subtitleTopToggle').checked;
    state.subtitlePosition = isTop ? 'top' : 'bottom';

    if (isTop) {
        elements.subtitleOverlay.classList.add('top');
    } else {
        elements.subtitleOverlay.classList.remove('top');
    }
}

// === Toast Notifications ===
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span>${getToastIcon(type)}</span>
        <span>${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('toast-fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function getToastIcon(type) {
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    return icons[type] || 'ℹ️';
}

// === Utility Functions ===
function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

function formatSpeed(bytesPerSecond) {
    if (bytesPerSecond < 1024) return bytesPerSecond.toFixed(0) + ' B/s';
    if (bytesPerSecond < 1024 * 1024) return (bytesPerSecond / 1024).toFixed(1) + ' KB/s';
    return (bytesPerSecond / (1024 * 1024)).toFixed(2) + ' MB/s';
}

function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) return 'Calculare...';
    
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function formatTimestamp(seconds) {
    if (isNaN(seconds) || !isFinite(seconds)) return '00:00:00,000';
    
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function downloadFile(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function getLanguageName(code) {
    const targetSelect = document.getElementById('targetLanguageSelect');
    const option = targetSelect?.querySelector(`option[value="${code}"]`);
    return option ? option.textContent : code;
}

function updateSubtitleLangSelect() {
    if (!elements.subtitleLangSelect) return;

    elements.subtitleLangSelect.innerHTML = '<option value="original">Original</option>';

    Object.keys(state.translations).forEach(lang => {
        const option = document.createElement('option');
        option.value = lang;
        option.textContent = getLanguageName(lang);
        elements.subtitleLangSelect.appendChild(option);
    });

    elements.subtitleLangSelect.style.display = 'inline-block';
    elements.subtitleLangSelect.value = state.displayLanguage;
}

function changeSubtitleLanguage(lang) {
    state.displayLanguage = lang;
    if (elements.mainVideoPlayer) {
        updateSubtitleDisplay(elements.mainVideoPlayer.currentTime);
    }
    updateFullText();
    renderSegments();
    renderTimeline();
}

// === Keyboard Shortcuts ===
document.addEventListener('keydown', (e) => {
    // Space for play/pause (doar când nu e focus pe un input)
    if (e.code === 'Space' && document.activeElement === document.body) {
        e.preventDefault();
        togglePlayPause();
    }
    
    // Ctrl+S for SRT export
    if (e.ctrlKey && e.code === 'KeyS') {
        e.preventDefault();
        exportSRT();
    }
    
    // Ctrl+D for DOCX export
    if (e.ctrlKey && e.code === 'KeyD') {
        e.preventDefault();
        showDOCXDialog();
    }
    
    // Escape to close modals
    if (e.code === 'Escape') {
        closeDOCXDialog();
    }
    
    // Săgeți pentru navigare între segmente
    if (e.code === 'ArrowUp' && state.activeSegment > 0) {
        e.preventDefault();
        seekToTime(state.segments[state.activeSegment - 1].start);
    }
    if (e.code === 'ArrowDown' && state.activeSegment < state.segments.length - 1) {
        e.preventDefault();
        seekToTime(state.segments[state.activeSegment + 1].start);
    }

    // Delete pentru ștergere segment selectat
    if (e.code === 'Delete' && state.selectedSegment !== null && document.activeElement === document.body) {
        e.preventDefault();
        deleteSegment(state.selectedSegment);
    }
});

// === Cleanup on page unload ===
window.addEventListener('beforeunload', () => {
    if (state.processingInterval) {
        clearInterval(state.processingInterval);
    }
    if (state.videoUrl) {
        URL.revokeObjectURL(state.videoUrl);
    }
});
// === Timeline Logic ===
function renderTimeline() {
    if (!elements.mainVideoPlayer || isNaN(elements.mainVideoPlayer.duration)) return;

    const duration = elements.mainVideoPlayer.duration;
    const pps = state.pixelsPerSecond * state.zoomLevel;
    const width = duration * pps;

    elements.timelineContent.style.width = width + 'px';
    elements.timelineRuler.style.width = width + 'px';

    // Render Ruler
    renderTimelineRuler(duration, pps);

    // Render Segments
    elements.timelineSegments.innerHTML = '';

    // Simple track management for overlapping segments
    const tracks = [];

    state.segments.forEach((segment, index) => {
        const startX = segment.start * pps;
        const endX = segment.end * pps;
        const segmentWidth = endX - startX;

        // Find a track that doesn't overlap
        let trackIndex = tracks.findIndex(trackEnd => trackEnd <= segment.start);
        if (trackIndex === -1) {
            trackIndex = tracks.length;
            tracks.push(segment.end);
        } else {
            tracks[trackIndex] = segment.end;
        }

        const block = document.createElement('div');
        block.className = 'timeline-segment-block';
        block.style.left = startX + 'px';
        block.style.width = segmentWidth + 'px';
        block.style.top = (trackIndex * 35 + 5) + 'px';
        block.textContent = segment.text;
        block.title = `${formatTimestamp(segment.start)} - ${formatTimestamp(segment.end)}\n${segment.text}`;

        if (state.selectedSegment === index) {
            block.classList.add('selected');
        }

        // Add text based on display language
        let displayText = segment.text;
        if (state.displayLanguage !== 'original' && state.translations[state.displayLanguage]) {
            displayText = state.translations[state.displayLanguage][index];
        }
        block.textContent = displayText;

        // Add handles
        const leftHandle = document.createElement('div');
        leftHandle.className = 'timeline-resize-handle left';
        const rightHandle = document.createElement('div');
        rightHandle.className = 'timeline-resize-handle right';
        const deleteBtn = document.createElement('div');
        deleteBtn.className = 'timeline-delete-btn';
        deleteBtn.innerHTML = '×';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteSegment(index);
        };

        block.appendChild(leftHandle);
        block.appendChild(rightHandle);
        block.appendChild(deleteBtn);

        // Interaction Events
        block.onmousedown = (e) => {
            if (e.button !== 0) return;

            e.stopPropagation(); // Prevent timelineContent playhead seek

            if (e.target.classList.contains('timeline-delete-btn')) return;

            state.selectedSegment = index;
            renderTimeline();
            renderSegments();

            // Only seek if it's a simple click (not a drag start yet, but we'll see)
            // Actually, always seek to start of segment when clicking it
            seekToTime(segment.start, false);

            state.isDragging = true;
            state.dragTarget = index;
            state.dragStartX = e.clientX;

            if (e.target.classList.contains('left')) {
                state.resizeType = 'start';
            } else if (e.target.classList.contains('right')) {
                state.resizeType = 'end';
            } else {
                state.resizeType = 'move';
                state.dragStartOffset = segment.start;
            }

            e.preventDefault();
        };

        elements.timelineSegments.appendChild(block);
    });

    // Adjust container height based on tracks
    elements.timelineContainer.style.height = Math.max(120, tracks.length * 35 + 20) + 'px';

    // Setup interactions once
    if (!window.timelineInited) {
        initTimelineInteractions();
        window.timelineInited = true;
    }
}

function initTimelineInteractions() {
    window.addEventListener('mousemove', handleTimelineMove);
    window.addEventListener('mouseup', handleTimelineUp);

    const timelineSection = document.getElementById('timelineSection');
    const timelineContainer = elements.timelineContainer;

    // Use addEventListener with { passive: false } for maximum reliability across browsers
    timelineSection.addEventListener('wheel', (e) => {
        // Block browser scroll for the whole section
        e.preventDefault();

        // Only handle zoom for vertical wheel
        if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            const contentRect = elements.timelineContent.getBoundingClientRect();
            const x = e.clientX - contentRect.left;

            const ppsBefore = state.pixelsPerSecond * state.zoomLevel;
            const timeAtCursor = x / ppsBefore;

            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;

            const oldZoom = state.zoomLevel;
            state.zoomLevel *= zoomFactor;
            state.zoomLevel = Math.max(0.1, Math.min(state.zoomLevel, 50));

            if (state.zoomLevel !== oldZoom) {
                elements.zoomLevel.textContent = Math.round(state.zoomLevel * 100) + '%';
                renderTimeline();

                const ppsAfter = state.pixelsPerSecond * state.zoomLevel;
                const newX = timeAtCursor * ppsAfter;
                elements.timelineContainer.scrollLeft += (newX - x);
            }
        }
    }, { passive: false });

    // Playhead dragging via the red triangle handle
    document.addEventListener('mousedown', (e) => {
        const handle = e.target.closest('#playheadHandle');
        if (handle) {
            e.preventDefault();
            e.stopPropagation();
            state.isDraggingPlayhead = true;
            document.body.style.cursor = 'grabbing';
        }
    });

    elements.timelineContent.addEventListener('mousedown', (e) => {
        // Middle button (wheel) panning
        if (e.button === 1) {
            e.preventDefault();
            state.isPanning = true;
            state.panStartX = e.clientX;
            state.panStartScroll = timelineContainer.scrollLeft;
            document.body.style.cursor = 'grabbing';
            return;
        }

        // Allow clicking on ruler or background to seek
        if (e.target.closest('.timeline-segment-block') ||
            e.target.closest('.timeline-resize-handle') ||
            e.target.closest('.timeline-delete-btn') ||
            e.target.closest('#playheadHandle')) return;

        state.isDraggingPlayhead = true;
        handleTimelineSeek(e);
        e.preventDefault();
    });

    // Prevent middle click autoscroll menu in some browsers
    elements.timelineContent.addEventListener('auxclick', (e) => {
        if (e.button === 1) e.preventDefault();
    });
}

function handleTimelineSeek(e) {
    if (!elements.mainVideoPlayer || isNaN(elements.mainVideoPlayer.duration)) return;

    const rect = elements.timelineContent.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pps = state.pixelsPerSecond * state.zoomLevel;
    const time = Math.max(0, Math.min(elements.mainVideoPlayer.duration, x / pps));

    state.selectedSegment = null;
    seekToTime(time, false);
}

function renderTimelineRuler(duration, pps) {
    elements.timelineRuler.innerHTML = '';

    // Determine interval based on zoom
    let interval = 5; // seconds
    if (pps > 100) interval = 1;
    if (pps < 20) interval = 10;
    if (pps < 5) interval = 30;

    for (let t = 0; t <= duration; t += interval) {
        const x = t * pps;
        const tick = document.createElement('div');
        tick.className = 'time-tick major';
        tick.style.left = x + 'px';

        const label = document.createElement('div');
        label.className = 'time-tick-label';
        label.style.left = x + 'px';
        label.textContent = formatTimeShort(t);

        elements.timelineRuler.appendChild(tick);
        elements.timelineRuler.appendChild(label);

        // Minor ticks
        if (interval >= 5) {
            const minorInterval = interval / 5;
            for (let mt = t + minorInterval; mt < t + interval && mt <= duration; mt += minorInterval) {
                const mx = mt * pps;
                const mTick = document.createElement('div');
                mTick.className = 'time-tick';
                mTick.style.left = mx + 'px';
                elements.timelineRuler.appendChild(mTick);
            }
        }
    }
}

function formatTimeShort(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function updateTimelinePlayhead(currentTime) {
    const pps = state.pixelsPerSecond * state.zoomLevel;
    const x = currentTime * pps;
    elements.timelinePlayhead.style.left = x + 'px';

    // Auto-scroll timeline if playhead goes out of view
    const container = elements.timelineContainer;
    const scrollLeft = container.scrollLeft;
    const width = container.clientWidth;

    if (x < scrollLeft || x > scrollLeft + width) {
        container.scrollLeft = x - width / 2;
    }
}

function zoomTimeline(factor) {
    state.zoomLevel *= factor;
    state.zoomLevel = Math.max(0.1, Math.min(state.zoomLevel, 10));
    elements.zoomLevel.textContent = Math.round(state.zoomLevel * 100) + '%';
    renderTimeline();
    updateTimelinePlayhead(elements.mainVideoPlayer.currentTime);
}

function handleTimelineMove(e) {
    if (state.isPanning) {
        const delta = e.clientX - state.panStartX;
        elements.timelineContainer.scrollLeft = state.panStartScroll - delta;
        return;
    }

    if (state.isDraggingPlayhead) {
        handleTimelineSeek(e);
        return;
    }

    if (!state.isDragging || state.dragTarget === null) return;

    const pps = state.pixelsPerSecond * state.zoomLevel;
    const dx = (e.clientX - state.dragStartX) / pps;
    const segment = state.segments[state.dragTarget];

    if (state.resizeType === 'move') {
        const duration = segment.end - segment.start;
        segment.start = Math.max(0, state.dragStartOffset + dx);
        segment.end = segment.start + duration;
    } else if (state.resizeType === 'start') {
        const newStart = Math.min(segment.end - 0.1, segment.start + dx);
        segment.start = Math.max(0, newStart);
        state.dragStartX = e.clientX;
    } else if (state.resizeType === 'end') {
        const newEnd = Math.max(segment.start + 0.1, segment.end + dx);
        segment.end = newEnd;
        state.dragStartX = e.clientX;
    }

    // Update visuals immediately without full re-render
    const block = elements.timelineSegments.children[state.dragTarget];
    if (block) {
        block.style.left = (segment.start * pps) + 'px';
        block.style.width = ((segment.end - segment.start) * pps) + 'px';
    }

    if (state.videoPlayer) {
        updateSubtitleDisplay(state.videoPlayer.currentTime);
        updateActiveSegment(state.videoPlayer.currentTime);
    }
}

function handleTimelineUp() {
    state.isDraggingPlayhead = false;
    state.isPanning = false;
    document.body.style.cursor = 'default';
    if (state.isDragging) {
        state.isDragging = false;
        state.dragTarget = null;
        renderTimeline();
        renderSegments(); // Update list
    }
}

function deleteSegment(index) {
    if (confirm('Sigur vrei să ștergi acest segment?')) {
        state.segments.splice(index, 1);
        // Also remove from translations if any
        Object.keys(state.translations).forEach(lang => {
            if (Array.isArray(state.translations[lang])) {
                state.translations[lang].splice(index, 1);
            }
        });

        state.selectedSegment = null;
        state.activeSegment = -1;

        // Refresh all UI components
        renderTimeline();
        renderSegments();
        displayTranslations();
        updateFullText();

        if (state.videoPlayer) {
            updateSubtitleDisplay(state.videoPlayer.currentTime);
            updateActiveSegment(state.videoPlayer.currentTime);
        }

        showToast('Segment șters cu succes', 'success');
    }
}
