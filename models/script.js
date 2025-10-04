document.addEventListener('DOMContentLoaded', function() {
    // Object Detection Code
    const detectButton = document.getElementById('detectButton');
    const imageInput = document.getElementById('imageInput');
    const uploadedImage = document.getElementById('uploadedImage');
    const detectionsContainer = document.getElementById('detections');
    const loadingIndicator = document.getElementById('loading');
    const detectionResults = document.getElementById('detection-results');
    const detectionList = document.getElementById('detection-list');
    
    detectButton.addEventListener('click', async function() {
        const file = imageInput.files[0];
        
        if (!file) {
            alert('Please select an image');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        detectionsContainer.innerHTML = '';
        detectionResults.style.display = 'none';
        
        // Display uploaded image
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Send to backend
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('http://localhost:5000/detect', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            drawDetections(data.detections);
            displayDetectionResults(data.detections);
        } catch (error) {
            console.error('Error:', error);
            alert(`Detection failed: ${error.message}\n\nPlease make sure the Flask server is running on http://localhost:5000`);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });
    
    function drawDetections(detections) {
        detectionsContainer.innerHTML = '';
        
        uploadedImage.onload = function() {
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
        };
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

    // Initialize chart
    const ctx = document.getElementById('densityChart').getContext('2d');
    const densityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High'],
            datasets: [{
                data: [30, 50, 20],
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

    // Simulate real-time updates
    function updateMetrics() {
        // Generate random values for demonstration
        const headCount = Math.floor(Math.random() * 100) + 10;
        let densityLevel;
        let densityClass;
        
        if (headCount < 30) {
            densityLevel = 'Low';
            densityClass = 'status-low';
        } else if (headCount < 60) {
            densityLevel = 'Medium';
            densityClass = 'status-medium';
        } else {
            densityLevel = 'High';
            densityClass = 'status-high';
        }
        
        // Update DOM elements
        document.getElementById('head-count').textContent = headCount;
        document.getElementById('density-level').textContent = densityLevel;
        
        // Update density badge
        const badge = document.getElementById('density-badge');
        badge.className = 'status-badge ' + densityClass;
        badge.innerHTML = '<i class="fas fa-users"></i> ' + densityLevel + ' Crowd';
        
        // Update chart
        let lowPercent, mediumPercent, highPercent;
        
        if (densityLevel === 'Low') {
            lowPercent = 80;
            mediumPercent = 15;
            highPercent = 5;
        } else if (densityLevel === 'Medium') {
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

    // Initial update and set interval
    updateMetrics();
    setInterval(updateMetrics, 5000); // Update every 5 seconds
});