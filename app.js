// app.js
import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const landmarksIndices = {
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
    LEFT_PINKY: 21,
    RIGHT_PINKY: 22,
    LEFT_THUMB: 19,
    RIGHT_THUMB: 20,
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
        console.log('Webcam enabled.');
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

async function predictWebcam() {
    console.log('Starting webcam prediction...');
    const drawingUtils = new DrawingUtils(canvasCtx);

    async function onFrame() {
        if (!webcamRunning) return;

        const now = performance.now();
        const elapsed = now - lastFrameTime;

        if (elapsed > 1000 / targetFPS) {
            lastFrameTime = now;

            const results = await poseLandmarker.detectForVideo(video, performance.now());

            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

            if (results.landmarks && results.landmarks[0]) {
                const landmarks = smoothLandmarks(results.landmarks[0], previousLandmarks);
                previousLandmarks = landmarks;

                drawingUtils.drawLandmarks(landmarks, { radius: 5 });
                drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);

                const leftHand = landmarks[landmarksIndices.LEFT_THUMB];
                const rightHand = landmarks[landmarksIndices.RIGHT_THUMB];

                if (leftHand && rightHand) {
                    drawShinai(leftHand, rightHand);
                }

                switch (currentStage) {
                    case STAGE_WEBCAM_ON:
                        if (leftHand && rightHand) {
                            startAnalysisButton.disabled = false;
                            updateMessage("Wrists detected", "green");
                        } else {
                            startAnalysisButton.disabled = true;
                            updateMessage("Wrists not detected", "red");
                        }
                        break;

                    case STAGE_READY_CHECK:
                        if (leftHand && rightHand) {
                            const currentTime = performance.now();
                            if (!initialWristPositions) {
                                initialWristPositions = { left: leftHand, right: rightHand };
                                stableStartTime = currentTime;
                            } else {
                                const leftMovement = distanceBetweenPoints(leftHand, initialWristPositions.left);
                                const rightMovement = distanceBetweenPoints(rightHand, initialWristPositions.right);

                                if (leftMovement < WRIST_STABILITY_THRESHOLD && rightMovement < WRIST_STABILITY_THRESHOLD) {
                                    const elapsedTime = currentTime - stableStartTime;
                                    updateMessage(`Hold still... ${(2 - (elapsedTime / 1000)).toFixed(1)}s`, "blue");
                                    if (elapsedTime >= 3000) {
                                        currentStage = STAGE_READY;
                                        initialWristPositions = { left: leftHand, right: rightHand };
                                        stableStartTime = null;
                                        updateMessage("Ready!", "green");
                                        startAnalysisButton.innerText = "ANALYSIS STARTED";
                                    }
                                } else {
                                    stableStartTime = currentTime;
                                    initialWristPositions = { left: leftHand, right: rightHand };
                                    updateMessage("Please hold still", "red");
                                }
                            }
                        } else {
                            stableStartTime = null;
                            initialWristPositions = null;
                            updateMessage("Wrists not detected", "red");
                        }
                        break;

                    case STAGE_READY:
                        updateMessage("Ready!", "green");

                        if (leftHand && rightHand) {
                            const leftMovement = distanceBetweenPoints(leftHand, initialWristPositions.left);
                            const rightMovement = distanceBetweenPoints(rightHand, initialWristPositions.right);

                            if (leftMovement >= SWING_START_THRESHOLD || rightMovement >= SWING_START_THRESHOLD) {
                                currentStage = STAGE_SWING;
                            }
                        } else {
                            currentStage = STAGE_READY_CHECK;
                            stableStartTime = null;
                            initialWristPositions = null;
                        }
                        break;

                    case STAGE_SWING:
                        calculateSwingAngle(landmarks);
                        break;

                    default:
                        break;
                }
            } else {
                if (currentStage === STAGE_READY_CHECK || currentStage === STAGE_READY) {
                    stableStartTime = null;
                    initialWristPositions = null;
                    updateMessage("Wrists not detected", "red");
                }
            }
        }

        requestAnimationFrame(onFrame);
    }

    onFrame();
}

function calculateSwingAngle(landmarks) {
    console.log('Calculating swing angle...');
    const leftHand = landmarks[landmarksIndices.LEFT_THUMB];
    const rightHand = landmarks[landmarksIndices.RIGHT_THUMB];

    if (!leftHand || !rightHand) return;

    const shinaiVector = {
        x: leftHand.x - rightHand.x,
        y: leftHand.y - rightHand.y,
        z: leftHand.z - rightHand.z
    };

    const magnitude = Math.sqrt(shinaiVector.x ** 2 + shinaiVector.y ** 2 + shinaiVector.z ** 2);
    const shinaiDirection = {
        x: shinaiVector.x / magnitude,
        y: shinaiVector.y / magnitude,
        z: shinaiVector.z / magnitude
    };

    const referenceVector = { x: 0, y: -1, z: 0 };

    const dotProduct =
        shinaiDirection.x * referenceVector.x +
        shinaiDirection.y * referenceVector.y +
        shinaiDirection.z * referenceVector.z;

    const angleRadians = Math.acos(dotProduct);
    const angleDegrees = (angleRadians * 180) / Math.PI;

    const cutType = angleDegrees > 45 ? 'Big Cut' : 'Small Cut';

    updateMessage(`Swing Angle: ${angleDegrees.toFixed(2)}°<br>Cut Type: ${cutType}`, "blue");

    drawShinai(leftHand, rightHand);
    currentStage = STAGE_WEBCAM_ON;
    startAnalysisButton.disabled = false;
    startAnalysisButton.innerText = "START ANALYSIS";
}

function drawShinai(leftHand, rightHand) {
    const startX = leftHand.x * canvasElement.width;
    const startY = leftHand.y * canvasElement.height;
    const endX = rightHand.x * canvasElement.width;
    const endY = rightHand.y * canvasElement.height;

    canvasCtx.beginPath();
    canvasCtx.moveTo(startX, startY);
    canvasCtx.lineTo(endX, endY);
    canvasCtx.strokeStyle = 'blue';
    canvasCtx.lineWidth = 4;
    canvasCtx.stroke();
}

function distanceBetweenPoints(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}