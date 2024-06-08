document.addEventListener('DOMContentLoaded', (event) => {
    const socket = io();

    socket.on('log', function (msg) {
        const logDiv = document.getElementById('log');
        const message = document.createElement('p');
        message.textContent = msg.data;
        logDiv.appendChild(message);
    });

    // Listen for clear_logs event to clear logs
    socket.on('clear_logs', function () {
        const logDiv = document.getElementById('log');
        logDiv.innerHTML = ''; // Clear the log element
    });

    // Listen for models_loaded event
    socket.on('models_loaded', function (msg) {
        const logDiv = document.getElementById('log');
        const message = document.createElement('p');
        message.textContent = msg.data;
        logDiv.appendChild(message);
    });

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
            //document.getElementById('result').textContent = JSON.stringify(result, null, 2);
        });
    }
});
