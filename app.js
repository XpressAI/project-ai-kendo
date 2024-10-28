import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@latest";
import { drawShinai } from './drawShinai.js';

let poseLandmarker = undefined;
let runningMode = "VIDEO";
let enableWebcamButton;
let startAnalysisButton;
let webcamRunning = false;
const messageDiv = document.getElementById("message");

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const STAGE_IDLE = 0;
const STAGE_WEBCAM_ON = 1;
const STAGE_READY_CHECK = 2;
const STAGE_READY = 3;
const STAGE_SWING = 4;
const STAGE_SHOW_RESULTS = 5;

let currentStage = STAGE_IDLE;

const WRIST_STABILITY_THRESHOLD = 0.8;
const SWING_START_THRESHOLD = 0.08;

let stableStartTime = null;
let initialWristPositions = null;

function updateMessage(text, color = 'black') {
    messageDiv.style.color = color;
    messageDiv.innerHTML = text;
}

async function createPoseLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 1
    });
}
createPoseLandmarker();

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    startAnalysisButton = document.getElementById("startAnalysisButton");
    enableWebcamButton.addEventListener("click", enableCam);
    startAnalysisButton.addEventListener("click", startAnalysis);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

async function enableCam() {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmarker not loaded yet.");
        return;
    }

    if (webcamRunning) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE WEBCAM";
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        currentStage = STAGE_IDLE;
        startAnalysisButton.disabled = true;
        startAnalysisButton.innerText = "START ANALYSIS";
        updateMessage("");
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE WEBCAM";
        currentStage = STAGE_WEBCAM_ON;

        const constraints = { video: true };
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
        });
        
        // Always enable the start analysis button
        startAnalysisButton.disabled = false;
        updateMessage("Webcam enabled. Click 'Start Analysis' when ready", "blue");
    }
}

function startAnalysis() {
    if (currentStage === STAGE_WEBCAM_ON || currentStage === STAGE_SHOW_RESULTS) {
        currentStage = STAGE_READY_CHECK;
        stableStartTime = null;
        initialWristPositions = null;
        startAnalysisButton.disabled = true;
        startAnalysisButton.innerText = "ANALYZING...";
    }
}

function calculateSwingAngle(landmarks) {
    const handCenters = drawShinai(landmarks, canvasElement, canvasCtx);
    if (!handCenters) return;

    const { leftHandCenter, rightHandCenter } = handCenters;

    const shinaiVector = {
        x: leftHandCenter.x - rightHandCenter.x,
        y: leftHandCenter.y - rightHandCenter.y
    };

    const magnitude = Math.sqrt(shinaiVector.x ** 2 + shinaiVector.y ** 2);
    const shinaiDirection = {
        x: shinaiVector.x / magnitude,
        y: shinaiVector.y / magnitude
    };

    const referenceVector = { x: 0, y: -1 };
    const dotProduct = shinaiDirection.x * referenceVector.x + shinaiDirection.y * referenceVector.y;
    const angleRadians = Math.acos(dotProduct);
    const angleDegrees = (angleRadians * 180) / Math.PI;

    const cutType = angleDegrees > 45 ? 'Big Cut' : 'Small Cut';
    
    currentStage = STAGE_SHOW_RESULTS;
    updateMessage(`Swing Angle: ${angleDegrees.toFixed(2)}°<br>Cut Type: ${cutType}<br><br>Click 'Start Analysis' to try again`, "blue");
    startAnalysisButton.disabled = false;
    startAnalysisButton.innerText = "START ANALYSIS";
    
    return angleDegrees;
}

function distanceBetweenPoints(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

async function predictWebcam() {
    if (!webcamRunning) return;

    const results = await poseLandmarker.detectForVideo(video, performance.now());

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks && results.landmarks[0]) {
        const landmarks = results.landmarks[0];
        const drawingUtils = new DrawingUtils(canvasCtx);
        drawingUtils.drawLandmarks(landmarks, { radius: 5 });
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);

        const handCenters = drawShinai(landmarks, canvasElement, canvasCtx);

        if (handCenters) {
            const { leftHandCenter, rightHandCenter } = handCenters;

            switch (currentStage) {
                case STAGE_WEBCAM_ON:
                    const leftWrist = landmarks[15];  // Left wrist landmark
                    const rightWrist = landmarks[16]; // Right wrist landmark

                    // Display message for hands but don't disable the button
                    if (leftWrist && rightWrist) {
                        if (leftWrist.visibility > 0.7 && rightWrist.visibility > 0.7) {
                            updateMessage("Hands detected", "green");
                        } else {
                            updateMessage(`Show both hands clearly for better accuracy`, "red");
                        }
                    } else {
                        updateMessage("Ensure both hands are visible", "red");
                    }
                    break;

                case STAGE_READY_CHECK:
                    if (!initialWristPositions) {
                        initialWristPositions = { left: leftHandCenter, right: rightHandCenter };
                        stableStartTime = performance.now();
                        updateMessage("Please hold still", "blue");
                    } else {
                        const leftMovement = distanceBetweenPoints(leftHandCenter, initialWristPositions.left);
                        const rightMovement = distanceBetweenPoints(rightHandCenter, initialWristPositions.right);

                        if (leftMovement < WRIST_STABILITY_THRESHOLD && rightMovement < WRIST_STABILITY_THRESHOLD) {
                            const elapsedTime = performance.now() - stableStartTime;
                            updateMessage(`Hold still... ${Math.max(0, (2 - (elapsedTime / 1000))).toFixed(1)}s`, "blue");
                            
                            if (elapsedTime >= 2000) {
                                currentStage = STAGE_READY;
                                initialWristPositions = { left: leftHandCenter, right: rightHandCenter };
                                stableStartTime = null;
                                updateMessage("Ready! Make your swing!", "green");
                            }
                        } else {
                            stableStartTime = performance.now();
                            initialWristPositions = { left: leftHandCenter, right: rightHandCenter };
                            updateMessage("Too much movement, please hold still", "red");
                        }
                    }
                    break;

                case STAGE_READY:
                    updateMessage("Ready! Make your swing!", "green");
                    const leftMovement = distanceBetweenPoints(leftHandCenter, initialWristPositions.left);
                    const rightMovement = distanceBetweenPoints(rightHandCenter, initialWristPositions.right);

                    if (leftMovement >= SWING_START_THRESHOLD || rightMovement >= SWING_START_THRESHOLD) {
                        currentStage = STAGE_SWING;
                        const angle = calculateSwingAngle(landmarks);
                        if (!angle || angle < 10) {
                            currentStage = STAGE_READY_CHECK;
                            updateMessage("Swing not detected. Please try again.", "red");
                            startAnalysisButton.disabled = false;
                            startAnalysisButton.innerText = "START ANALYSIS";
                        }
                    }
                    break;
            }
        } else {
            startAnalysisButton.disabled = true;
            updateMessage("Hands not detected", "red");
        }
    } else {
        startAnalysisButton.disabled = true;
        updateMessage("No pose detected", "red");
    }

    requestAnimationFrame(predictWebcam);
}