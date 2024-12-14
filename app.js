// app.js

import { PoseProcessor } from './PoseProcessor.js';

const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const messageDiv = document.getElementById("message");
const enableWebcamButton = document.getElementById("webcamButton");
const startAnalysisButton = document.getElementById("startAnalysisButton");

const poseProcessor = new PoseProcessor(canvasElement, videoElement, messageDiv);

// Initialize the Material Components Web buttons (if using MDC)
if (mdc && mdc.ripple) {
    mdc.ripple.MDCRipple.attachTo(enableWebcamButton);
    mdc.ripple.MDCRipple.attachTo(startAnalysisButton);
}

// Button event listeners
enableWebcamButton.addEventListener("click", () => {
    const labelSpan = enableWebcamButton.querySelector('.mdc-button__label');
    if (poseProcessor.webcamRunning) {
        poseProcessor.stopWebcamStream();
        labelSpan.innerText = "ENABLE WEBCAM";
        startAnalysisButton.disabled = true;
    } else {
        poseProcessor.startWebcamStream();
        labelSpan.innerText = "DISABLE WEBCAM";
        startAnalysisButton.disabled = false;
    }
});

startAnalysisButton.addEventListener("click", () => {
    poseProcessor.startAnalysis();
});
