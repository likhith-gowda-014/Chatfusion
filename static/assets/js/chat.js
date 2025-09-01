let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            // Show feedback for recording
            document.getElementById('responseAudio').textContent = "Recording...";

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });  // Use 'audio/wav' for better compatibility
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');
                formData.append('voice_id', document.getElementById('voice_id').value);
                formData.append('prompt-template', document.getElementById('prompt-template').value);  // Added prompt-template

                fetch('/upload-audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    const audio = document.getElementById('responseAudio');
                    audio.src = `/get-audio/${data.response_audio.split('/').pop()}`;  // Ensure file path is correct
                    audio.play();
                    document.getElementById('responseAudio').textContent = "Playback";  // Reset text content
                })
                .catch(error => {
                    console.error("Error while sending audio: ", error);
                    document.getElementById('responseAudio').textContent = "Error, please try again.";
                });
            };
        })
        .catch(error => {
            console.error("Error accessing microphone: ", error);
            alert("Please grant microphone access to use this feature.");
        });
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
    }
}
