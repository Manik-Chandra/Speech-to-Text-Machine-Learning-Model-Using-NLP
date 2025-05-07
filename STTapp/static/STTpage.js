// DOM Elements
const recordBtn = document.getElementById('record-btn');
const audioPlayer = document.getElementById('audio');
const transcriptionBox = document.getElementById('transcription');
const languageSelect = document.getElementById('language-select');

// Audio recording variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let audioBlob = null;

// API endpoint - update this to match your Django URL
const API_ENDPOINT = '/speech-recognition/';

// Record button click handler
recordBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
  if (!isRecording) {
    await startRecording();
    recordBtn.classList.add('recording');
  } else {
    stopRecording();
    recordBtn.classList.remove('recording');
  }
}

async function startRecording() {
  try {
    isRecording = true;
    audioChunks = [];
    transcriptionBox.value = '';
    audioPlayer.src = '';

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      audioPlayer.src = audioUrl; // Set audio source immediately after recording
      await processRecording(audioBlob);
    };

    mediaRecorder.start();

    // Auto-stop after 30 seconds
    setTimeout(() => {
      if (isRecording) toggleRecording();
    }, 30000);

  } catch (error) {
    console.error('Recording error:', error);
    transcriptionBox.value = 'Error: Microphone access denied';
    resetRecording();
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
  }
}

function resetRecording() {
  isRecording = false;
  recordBtn.classList.remove('recording');
}

async function processRecording(audioBlob) {
  const transcriptionBox = document.getElementById('transcription');
  transcriptionBox.value = 'Processing...';

  const audioPlayer = document.getElementById('audio');
  const audioUrl = URL.createObjectURL(audioBlob);
  audioPlayer.src = audioUrl;

  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');

  // Send selected language to Django backend
  const selectedLanguage = languageSelect.value;
  formData.append('language', selectedLanguage);

  try {
    const response = await fetch('/speech-recognition/', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    if (data.status === 'error') {
      transcriptionBox.value = data.message;
    } else {
      transcriptionBox.value = data.transcription;
    }

  } catch (error) {
    transcriptionBox.value = 'Error:' + error.message;
    console.error('Error:', error);
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

// Helper function to get CSRF token
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}
