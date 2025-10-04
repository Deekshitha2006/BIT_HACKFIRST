// Temple data - updated to match the temple selection page
        const temples = {
            iskon: {
                name: "ISKCON Temple Bangalore",
                location: "Rajajinagar, Bangalore"
            },
            dharmasthala: {
                name: "Dharmasthala Manjunatha",
                location: "Dharmasthala, Dakshina Kannada"
            },
            kukke: {
                name: "Kukke Subrahmanya",
                location: "Subrahmanya, Dakshina Kannada"
            },
            chamundi: {
                name: "Chamundeshwari Temple",
                location: "Chamundi Hills, Mysore"
            },
            udupi: {
                name: "Udupi Sri Krishna",
                location: "Udupi, Karnataka"
            },
            murudeshwara: {
                name: "Murudeshwara Temple",
                location: "Murudeshwar, Uttara Kannada"
            }
        };

        document.addEventListener('DOMContentLoaded', function() {
            // Get temple ID from URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            const templeId = urlParams.get('id');
            
            // Set temple information
            if (templeId && temples[templeId]) {
                document.getElementById('temple-name').textContent = temples[templeId].name;
                document.getElementById('temple-location').textContent = temples[templeId].location;
            } else {
                // Default to ISKCON if no ID or invalid ID
                document.getElementById('temple-name').textContent = temples.iskon.name;
                document.getElementById('temple-location').textContent = temples.iskon.location;
            }
            
            // Object Detection Code
            const detectButton = document.getElementById('detectButton');
            const imageInput = document.getElementById('imageInput');
            const uploadedImage = document.getElementById('uploadedImage');
            const detectionsContainer = document.getElementById('detections');
            const loadingIndicator = document.getElementById('loading');
            const loadingText = document.getElementById('loading-text');
            const detectionResults = document.getElementById('detection-results');
            const detectionList = document.getElementById('detection-list');
            const detectCrowdButton = document.getElementById('detectCrowdButton');
            const crowdMetrics = document.getElementById('crowd-metrics');
            const peopleCountElement = document.getElementById('people-count');
            const crowdLevelElement = document.getElementById('crowd-level');
            const crowdDensityElement = document.getElementById('crowd-density');
            
            // Initialize chart
            const ctx = document.getElementById('densityChart').getContext('2d');
            const densityChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Low', 'Medium', 'High'],
                    datasets: [{
                        data: [100, 0, 0], // Initially all low
                        backgroundColor: [
                            '#2ecc71', // Green for low
                            '#f39c12', // Orange for medium
                            '#e74c3c'  // Red for high
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: {
                                    size: 14
                                },
                                padding: 20
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return context.label + ': ' + context.raw + '%';
                                }
                            }
                        }
                    },
                    cutout: '70%'
                }
            });
            
            detectButton.addEventListener('click', async function() {
                const file = imageInput.files[0];
                
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                // Show loading indicator
                loadingIndicator.style.display = 'flex';
                loadingText.textContent = 'Processing image...';
                detectionsContainer.innerHTML = '';
                detectionResults.style.display = 'none';
                crowdMetrics.style.display = 'none';
                uploadedImage.style.display = 'block';
                
                // Display uploaded image
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImage.src = e.target.result;
                };
                reader.readAsDataURL(file);
                
                // Send to backend with timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('http://localhost:5000/detect', {
                        method: 'POST',
                        body: formData,
                        signal: controller.signal
                    });
                    
                    clearTimeout(timeoutId);
                    
                    if (!response.ok) {
                        throw new Error(`Server responded with status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    drawDetections(data.detections);
                    displayDetectionResults(data.detections);
                    
                    // Update dashboard with real data
                    if (data.people_count !== undefined) {
                        updateDashboard(data.people_count, "Medium"); // Default to Medium for standard detection
                    }
                } catch (error) {
                    if (error.name === 'AbortError') {
                        alert('Processing timed out. Please try with a smaller image.');
                    } else {
                        console.error('Error:', error);
                        alert(`Detection failed: ${error.message}\n\nPlease make sure the Flask server is running on http://localhost:5000`);
                    }
                } finally {
                    loadingIndicator.style.display = 'none';
                }
            });
            
            detectCrowdButton.addEventListener('click', async function() {
                const file = imageInput.files[0];
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                // Reset containers
                detectionsContainer.innerHTML = '';
                uploadedImage.style.display = 'block';
                detectionResults.style.display = 'none';
                crowdMetrics.style.display = 'none';
                
                // Show loading indicator
                loadingIndicator.style.display = 'flex';
                loadingText.textContent = 'Processing crowd detection (this may take up to 60 seconds)...';
                
                // Display uploaded image
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImage.src = e.target.result;
                };
                reader.readAsDataURL(file);
                
                // Send to backend with timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('http://localhost:5000/detect_crowd', {
                        method: 'POST',
                        body: formData,
                        signal: controller.signal
                    });
                    
                    clearTimeout(timeoutId);
                    
                    if (!response.ok) {
                        throw new Error(`Server responded with status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    drawDetections(data.detections);
                    displayDetectionResults(data.detections);
                    
                    // Show enhanced crowd metrics
                    peopleCountElement.textContent = data.people_count;
                    crowdLevelElement.textContent = data.crowd_level;
                    crowdDensityElement.textContent = data.crowd_density;
                    crowdMetrics.style.display = 'block';
                    
                    // Update the dashboard with real crowd data
                    updateDashboard(data.people_count, data.crowd_level);
                    
                } catch (error) {
                    if (error.name === 'AbortError') {
                        alert('Crowd detection timed out. Please try with a smaller image or use standard detection.');
                    } else {
                        console.error('Error:', error);
                        alert(`Detection failed: ${error.message}\n\nPlease make sure the Flask server is running on http://localhost:5000`);
                    }
                } finally {
                    loadingIndicator.style.display = 'none';
                }
            });
            
            function updateDashboard(peopleCount, crowdLevel) {
                // Update the dashboard metrics
                document.getElementById('head-count').textContent = peopleCount;
                document.getElementById('density-level').textContent = crowdLevel;
                
                // Update density badge
                const badge = document.getElementById('density-badge');
                const statusText = document.getElementById('density-status');
                
                let densityClass;
                if (crowdLevel === 'Low') {
                    densityClass = 'status-low';
                } else if (crowdLevel === 'Medium') {
                    densityClass = 'status-medium';
                } else {
                    densityClass = 'status-high';
                }
                
                badge.className = 'status-badge ' + densityClass;
                statusText.textContent = crowdLevel + ' Crowd';
                
                // Update the chart
                let lowPercent, mediumPercent, highPercent;
                
                if (crowdLevel === 'Low') {
                    lowPercent = 80;
                    mediumPercent = 15;
                    highPercent = 5;
                } else if (crowdLevel === 'Medium') {
                    lowPercent = 30;
                    mediumPercent = 60;
                    highPercent = 10;
                } else {
                    lowPercent = 10;
                    mediumPercent = 20;
                    highPercent = 70;
                }
                
                densityChart.data.datasets[0].data = [lowPercent, mediumPercent, highPercent];
                densityChart.update();
            }
            
            function drawDetections(detections) {
                detectionsContainer.innerHTML = '';
                
                if (uploadedImage.complete) {
                    // Image is already loaded, draw immediately
                    drawBoundingBoxes(detections);
                } else {
                    // Wait for image to load
                    uploadedImage.onload = function() {
                        drawBoundingBoxes(detections);
                    };
                }
            }
            
            function drawBoundingBoxes(detections) {
                const scaleX = uploadedImage.clientWidth / uploadedImage.naturalWidth;
                const scaleY = uploadedImage.clientHeight / uploadedImage.naturalHeight;
                
                detections.forEach(detection => {
                    const [x1, y1, x2, y2] = detection.bbox;
                    
                    // Create bounding box
                    const box = document.createElement('div');
                    box.className = 'detection-box';
                    box.style.left = `${x1 * scaleX}px`;
                    box.style.top = `${y1 * scaleY}px`;
                    box.style.width = `${(x2 - x1) * scaleX}px`;
                    box.style.height = `${(y2 - y1) * scaleY}px`;
                    
                    // Create label
                    const label = document.createElement('div');
                    label.className = 'label';
                    label.textContent = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
                    label.style.left = `${x1 * scaleX}px`;
                    label.style.top = `${y1 * scaleY - 20}px`;
                    
                    detectionsContainer.appendChild(box);
                    detectionsContainer.appendChild(label);
                });
            }
            
            function displayDetectionResults(detections) {
                detectionList.innerHTML = '';
                
                if (detections.length === 0) {
                    detectionList.innerHTML = '<div class="detection-item">No objects detected</div>';
                } else {
                    // Count objects by class
                    const classCounts = {};
                    detections.forEach(detection => {
                        if (!classCounts[detection.class]) {
                            classCounts[detection.class] = {
                                count: 0,
                                confidences: []
                            };
                        }
                        classCounts[detection.class].count++;
                        classCounts[detection.class].confidences.push(detection.confidence);
                    });
                    
                    // Display results
                    for (const className in classCounts) {
                        const item = document.createElement('div');
                        item.className = 'detection-item';
                        
                        const avgConfidence = classCounts[className].confidences.reduce((a, b) => a + b, 0) / classCounts[className].confidences.length;
                        
                        item.innerHTML = `
                            <span class="detection-class">${className}: ${classCounts[className].count}</span>
                            <span class="detection-confidence">${(avgConfidence * 100).toFixed(1)}% confidence</span>
                        `;
                        
                        detectionList.appendChild(item);
                    }
                }
                
                detectionResults.style.display = 'block';
            }
        });

