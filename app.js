// app.js
import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const landmarksIndices = {
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
    LEFT_INDEX: 19,
    RIGHT_INDEX: 20,
    LEFT_PINKY: 17,
    RIGHT_PINKY: 18,
    LEFT_THUMB: 21,
    RIGHT_THUMB: 22,
};

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

let currentStage = STAGE_IDLE;

const WRIST_STABILITY_THRESHOLD = 0.02;
const SWING_START_THRESHOLD = 0.05;

let stableStartTime = null;
let initialWristPositions = null;
let lastFrameTime = 0;
const targetFPS = 15;
const smoothFactor = 0.8;
let previousLandmarks = null;

function updateMessage(text, color = 'black') {
    messageDiv.style.color = color;
    messageDiv.innerHTML = text;
}

async function createPoseLandmarker() {
    console.log('Creating pose landmarker...');
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 1
    });
    console.log('Pose landmarker created.');
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

function drawShinai(landmarks) {
    // Check if all required landmarks are present
    const requiredIndices = [
        landmarksIndices.LEFT_THUMB,
        landmarksIndices.LEFT_INDEX,
        landmarksIndices.LEFT_PINKY,
        landmarksIndices.RIGHT_THUMB,
        landmarksIndices.RIGHT_INDEX,
        landmarksIndices.RIGHT_PINKY
    ];

    const allLandmarksPresent = requiredIndices.every(index => 
        landmarks[index] && 
        landmarks[index].x !== undefined && 
        landmarks[index].y !== undefined
    );

    if (!allLandmarksPresent) {
        return;
    }

    // Get all the landmarks for both hands
    const leftHandPoints = {
        thumb: {
            x: landmarks[landmarksIndices.LEFT_THUMB].x * canvasElement.width,
            y: landmarks[landmarksIndices.LEFT_THUMB].y * canvasElement.height
        },
        index: {
            x: landmarks[landmarksIndices.LEFT_INDEX].x * canvasElement.width,
            y: landmarks[landmarksIndices.LEFT_INDEX].y * canvasElement.height
        },
        pinky: {
            x: landmarks[landmarksIndices.LEFT_PINKY].x * canvasElement.width,
            y: landmarks[landmarksIndices.LEFT_PINKY].y * canvasElement.height
        }
    };

    const rightHandPoints = {
        thumb: {
            x: landmarks[landmarksIndices.RIGHT_THUMB].x * canvasElement.width,
            y: landmarks[landmarksIndices.RIGHT_THUMB].y * canvasElement.height
        },
        index: {
            x: landmarks[landmarksIndices.RIGHT_INDEX].x * canvasElement.width,
            y: landmarks[landmarksIndices.RIGHT_INDEX].y * canvasElement.height
        },
        pinky: {
            x: landmarks[landmarksIndices.RIGHT_PINKY].x * canvasElement.width,
            y: landmarks[landmarksIndices.RIGHT_PINKY].y * canvasElement.height
        }
    };

    // Draw left hand grip
    canvasCtx.beginPath();
    canvasCtx.moveTo(leftHandPoints.thumb.x, leftHandPoints.thumb.y);
    canvasCtx.lineTo(leftHandPoints.index.x, leftHandPoints.index.y);
    canvasCtx.lineTo(leftHandPoints.pinky.x, leftHandPoints.pinky.y);
    canvasCtx.lineTo(leftHandPoints.thumb.x, leftHandPoints.thumb.y);
    canvasCtx.strokeStyle = 'red';
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();

    // Draw right hand grip
    canvasCtx.beginPath();
    canvasCtx.moveTo(rightHandPoints.thumb.x, rightHandPoints.thumb.y);
    canvasCtx.lineTo(rightHandPoints.index.x, rightHandPoints.index.y);
    canvasCtx.lineTo(rightHandPoints.pinky.x, rightHandPoints.pinky.y);
    canvasCtx.lineTo(rightHandPoints.thumb.x, rightHandPoints.thumb.y);
    canvasCtx.strokeStyle = 'blue';
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();

    // Draw connecting line between hands (shinai)
    canvasCtx.beginPath();
    // Calculate center points of each hand triangle
    const leftHandCenter = {
        x: (leftHandPoints.thumb.x + leftHandPoints.index.x + leftHandPoints.pinky.x) / 3,
        y: (leftHandPoints.thumb.y + leftHandPoints.index.y + leftHandPoints.pinky.y) / 3
    };
    const rightHandCenter = {
        x: (rightHandPoints.thumb.x + rightHandPoints.index.x + rightHandPoints.pinky.x) / 3,
        y: (rightHandPoints.thumb.y + rightHandPoints.index.y + rightHandPoints.pinky.y) / 3
    };
    
    // Draw the shinai line between the centers
    canvasCtx.moveTo(leftHandCenter.x, leftHandCenter.y);
    canvasCtx.lineTo(rightHandCenter.x, rightHandCenter.y);
    canvasCtx.strokeStyle = 'purple';
    canvasCtx.lineWidth = 4;
    canvasCtx.stroke();

    return { leftHandCenter, rightHandCenter };
}

async function enableCam() {
  console.log('Enabling/disabling webcam...');
  if (!poseLandmarker) {
        console.log("Wait! poseLandmarker not loaded yet.");
        return;
    }

    if (webcamRunning) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE WEBCAM";
        console.log('Webcam disabled.');
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
            console.log('Stream received.');
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                console.log('Video data loaded.');
                predictWebcam();
            });
        }).catch(function(err) {
            console.error('Error accessing the webcam: ' + err);
        });
    }
}

function startAnalysis() {
    if (currentStage === STAGE_WEBCAM_ON) {
        currentStage = STAGE_READY_CHECK;
        stableStartTime = null;
        initialWristPositions = null;
        startAnalysisButton.disabled = true;
        startAnalysisButton.innerText = "ANALYZING...";
    }
}

function smoothLandmarks(newLandmarks, previousLandmarks) {
    if (!previousLandmarks) return newLandmarks;

    return newLandmarks.map((newLandmark, i) => ({
        x: smoothFactor * previousLandmarks[i].x + (1 - smoothFactor) * newLandmark.x,
        y: smoothFactor * previousLandmarks[i].y + (1 - smoothFactor) * newLandmark.y,
        z: smoothFactor * previousLandmarks[i].z + (1 - smoothFactor) * newLandmark.z
    }));
}

function calculateSwingAngle(landmarks) {
    const handCenters = drawShinai(landmarks);
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
    updateMessage(`Swing Angle: ${angleDegrees.toFixed(2)}°<br>Cut Type: ${cutType}`, "blue");

    currentStage = STAGE_WEBCAM_ON;
    startAnalysisButton.disabled = false;
    startAnalysisButton.innerText = "START ANALYSIS";
}

function distanceBetweenPoints(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

async function predictWebcam() {
    if (!webcamRunning) return;

    const now = performance.now();
    const elapsed = now - lastFrameTime;

    if (elapsed > 1000 / targetFPS) {
        lastFrameTime = now;

        const results = await poseLandmarker.detectForVideo(video, now);

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

        if (results.landmarks && results.landmarks[0]) {
            const landmarks = smoothLandmarks(results.landmarks[0], previousLandmarks);
            previousLandmarks = landmarks;

            const drawingUtils = new DrawingUtils(canvasCtx);
            drawingUtils.drawLandmarks(landmarks, { radius: 5 });
            drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);

            const handCenters = drawShinai(landmarks);
            if (handCenters) {
                const { leftHandCenter, rightHandCenter } = handCenters;

                switch (currentStage) {
                    case STAGE_WEBCAM_ON:
                        startAnalysisButton.disabled = false;
                        updateMessage("Hands detected", "green");
                        break;

                    case STAGE_READY_CHECK:
                        if (!initialWristPositions) {
                            initialWristPositions = { left: leftHandCenter, right: rightHandCenter };
                            stableStartTime = now;
                        } else {
                            const leftMovement = distanceBetweenPoints(leftHandCenter, initialWristPositions.left);
                            const rightMovement = distanceBetweenPoints(rightHandCenter, initialWristPositions.right);

                            if (leftMovement < WRIST_STABILITY_THRESHOLD && rightMovement < WRIST_STABILITY_THRESHOLD) {
                                const elapsedTime = now - stableStartTime;
                                updateMessage(`Hold still... ${(2 - (elapsedTime / 1000)).toFixed(1)}s`, "blue");
                                if (elapsedTime >= 2000) {
                                    currentStage = STAGE_READY;
                                    initialWristPositions = { left: leftHandCenter, right: rightHandCenter };
                                    stableStartTime = null;
                                    updateMessage("Ready!", "green");
                                }
                            } else {
                                stableStartTime = now;
                                initialWristPositions = { left: leftHandCenter, right: rightHandCenter };
                                updateMessage("Please hold still", "red");
                            }
                        }
                        break;

                    case STAGE_READY:
                        updateMessage("Ready!", "green");
                        const leftMovement = distanceBetweenPoints(leftHandCenter, initialWristPositions.left);
                        const rightMovement = distanceBetweenPoints(rightHandCenter, initialWristPositions.right);

                        if (leftMovement >= SWING_START_THRESHOLD || rightMovement >= SWING_START_THRESHOLD) {
                            currentStage = STAGE_SWING;
                            calculateSwingAngle(landmarks);
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
    }

    requestAnimationFrame(predictWebcam);
}