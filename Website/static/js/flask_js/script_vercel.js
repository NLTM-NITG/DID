document.addEventListener('DOMContentLoaded', (event) => {

    async function fetchLogs() {
        const response = await fetch('/logs');
        const logs = await response.json();
        const logDiv = document.getElementById('log');
        logDiv.innerHTML = ''; // Clear previous logs
        logs.forEach(log => {
            const message = document.createElement('p');
            message.textContent = log;
            logDiv.appendChild(message);
        });
    }

    // Add event listener for replay button
    const replayBtn = document.getElementById('replay-btn');
    if (replayBtn) {
        replayBtn.addEventListener('click', async function () {
            const fileInput = document.getElementById('file-upload');
            const file = fileInput.files[0];
            if (file) {
                const audio = new Audio();
                audio.src = URL.createObjectURL(file);
                audio.play();
            }
        });
    }

    const fileUpload = document.getElementById('file-upload');
    if (fileUpload) {
        fileUpload.addEventListener('change', function () {
            const fileInput = document.getElementById('file-upload');
            const fileName = fileInput.files[0].name;
            document.getElementById('file-name').textContent = 'Uploaded File: ' + fileName;
        });
    }

    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', async function (event) {
            event.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/extract_features', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            // document.getElementById('result').textContent = JSON.stringify(result, null, 2);
        });
    }

    setInterval(fetchLogs, 1000);  // Fetch logs every second

});
