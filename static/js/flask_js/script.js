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
        try {
            const response = await fetch('/extract_features', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                // If response is not OK (status code other than 2xx), throw an error
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const result = await response.json();
            // document.getElementById('result').textContent = JSON.stringify(result, null, 2);
        } catch (error) {
            console.error('Error:', error);
            // Proceed with normal processing for other errors
        }
    });
}

// Display pop-up for any kind of error onclick of the button
document.addEventListener('click', function(event) {
    const target = event.target;
    if (target.tagName === 'BUTTON' && target.textContent === 'Okay') {
        displaySocketError();
    }
});

function displaySocketError() {
    // Create a pop-up container
    const popUpContainer = document.createElement('div');
    popUpContainer.classList.add('popup-container');

    // Create a pop-up message
    const popUpMessage = document.createElement('div');
    popUpMessage.classList.add('popup');
    popUpMessage.textContent = 'Server cannot be reached';

    // Create a button
    const button = document.createElement('button');
    button.textContent = 'Okay';

    // Add click event listener to the button
    button.addEventListener('click', function () {
        // Remove the pop-up container when the button is clicked
        document.body.removeChild(popUpContainer);
    });

    // Append the button to the pop-up message
    popUpMessage.appendChild(button);

    // Append the pop-up message to the pop-up container
    popUpContainer.appendChild(popUpMessage);

    // Append the pop-up container to the body
    document.body.appendChild(popUpContainer);
}

});
