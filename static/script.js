document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const fillBenignBtn = document.getElementById('fillBenignBtn');
    const fillMalignantBtn = document.getElementById('fillMalignantBtn');
    const clearBtn = document.getElementById('clearBtn');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = submitBtn.querySelector('span');
    const spinner = submitBtn.querySelector('.spinner');
    
    const resultCard = document.getElementById('resultCard');
    const predictionBadge = document.getElementById('predictionBadge');
    const probabilityText = document.getElementById('probabilityText');

    // Known values from data.csv for a typical Benign case
    const benignData = [
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 
        0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 
        15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ];

    // Known values from data.csv for a typical Malignant case
    const malignantData = [
        17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ];

    function fillForm(dataArray) {
        if (!window.FEATURE_NAMES) return;
        window.FEATURE_NAMES.forEach((feature, index) => {
            const input = document.getElementById(feature);
            if (input && dataArray[index] !== undefined) {
                // Randomize slightly so it doesn't look completely static every time
                const variation = dataArray[index] * (1 + (Math.random() * 0.04 - 0.02)); 
                input.value = Number(variation.toPrecision(6));
            }
        });
        
        // Hide result card if showing
        resultCard.classList.remove('visible');
        setTimeout(() => resultCard.classList.add('hidden'), 500);
    }

    fillBenignBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fillForm(benignData);
    });

    fillMalignantBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fillForm(malignantData);
    });

    clearBtn.addEventListener('click', (e) => {
        e.preventDefault();
        form.reset();
        resultCard.classList.remove('visible');
        setTimeout(() => resultCard.classList.add('hidden'), 500);
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        submitText.style.display = 'none';
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;
        resultCard.classList.remove('visible');
        
        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // Update UI with results
                predictionBadge.textContent = result.prediction;
                probabilityText.textContent = result.probability;
                
                // Set styling based on malignancy
                if (result.is_malignant) {
                    predictionBadge.className = 'badge malignant';
                    predictionBadge.innerHTML = `⚠️ ${result.prediction}`;
                } else {
                    predictionBadge.className = 'badge benign';
                    predictionBadge.innerHTML = `✅ ${result.prediction}`;
                }
                
                // Show result card
                resultCard.classList.remove('hidden');
                // Trigger reflow
                void resultCard.offsetWidth;
                resultCard.classList.add('visible');
                
                // Scroll to result smoothly
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            console.error('Error during prediction:', error);
            alert('An error occurred while connecting to the server.');
        } finally {
            // Restore button state
            submitText.style.display = 'inline';
            spinner.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
});
