// PoseProcessor.js

import { PoseLandmarker, DrawingUtils, FilesetResolver } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.18";
import { drawShinai } from './drawShinai.js';

export class PoseProcessor {
    constructor(canvasElement, videoElement, messageElement) {
        this.canvasElement = canvasElement;
        this.canvasCtx = canvasElement.getContext("2d");
        this.videoElement = videoElement;
        this.messageElement = messageElement;

        this.poseLandmarker = null;
        this.currentStage = this.STAGES.IDLE;
        this.stableStartTime = null;
        this.initialHandPositions = null;
        this.webcamRunning = false;
        this.processingVideoFile = false;

        // Adjust thresholds based on pixel measurements
        this.HAND_STABILITY_THRESHOLD = 10; // pixels
        this.SWING_START_THRESHOLD = 30;    // pixels

        // Initialize pose landmarker
        this.createPoseLandmarker();
    }

    // Define stages as constants
    STAGES = {
        IDLE: 0,
        WEBCAM_ON: 1,
        READY_CHECK: 2,
        READY: 3,
        SWING: 4,
        SHOW_RESULTS: 5,
    };

    async createPoseLandmarker() {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });
    }

    updateMessage(text, color = 'black') {
        this.messageElement.style.color = color;
        this.messageElement.innerHTML = text;
    }

    startWebcamStream() {
        const constraints = { video: true };
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            this.videoElement.srcObject = stream;
            this.videoElement.play();
            this.webcamRunning = true;
            this.processingVideoFile = false;
            this.currentStage = this.STAGES.WEBCAM_ON;
            this.updateMessage("Webcam enabled. Click 'Start Analysis' when ready", "blue");
            this.videoElement.addEventListener("loadeddata", () => this.processVideo());
        });
    }

    stopWebcamStream() {
        if (this.videoElement.srcObject) {
            this.videoElement.srcObject.getTracks().forEach(track => track.stop());
        }
        this.webcamRunning = false;
        this.currentStage = this.STAGES.IDLE;
        this.updateMessage("");
    }

    startVideoFileProcessing(file) {
        const fileURL = URL.createObjectURL(file);
        this.videoElement.src = fileURL;
        this.videoElement.play();
        this.webcamRunning = true;
        this.processingVideoFile = true;
        this.currentStage = this.STAGES.WEBCAM_ON;
        this.updateMessage("Video loaded. Click 'Start Analysis' when ready", "blue");
        this.videoElement.addEventListener("loadeddata", () => this.processVideo());
    }

    stopVideoFileProcessing() {
        this.videoElement.pause();
        this.videoElement.src = '';
        this.webcamRunning = false;
        this.processingVideoFile = false;
        this.currentStage = this.STAGES.IDLE;
        this.updateMessage("");
    }

    startAnalysis() {
        if (this.currentStage === this.STAGES.WEBCAM_ON || this.currentStage === this.STAGES.SHOW_RESULTS) {
            this.currentStage = this.STAGES.READY_CHECK;
            this.stableStartTime = null;
            this.initialHandPositions = null;
            this.updateMessage("Analysis started. Please hold still.", "blue");
        }
    }

    async processVideo() {
        if (!this.webcamRunning) return;

        // Check if the video has ended when processing a video file
        if (this.processingVideoFile && this.videoElement.ended) {
            this.updateMessage("Video playback ended.", "blue");
            this.stopVideoFileProcessing();
            return;
        }

        const results = await this.poseLandmarker.detectForVideo(this.videoElement, performance.now());

        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        this.canvasCtx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);

        if (results.landmarks && results.landmarks[0]) {
            const landmarks = results.landmarks[0];

            // Draw pose landmarks
            const drawingUtils = new DrawingUtils(this.canvasCtx);
            drawingUtils.drawLandmarks(landmarks, { radius: 3 });
            drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);

            // Draw shinai and get hand centers
            const handCenters = drawShinai(landmarks, this.canvasElement, this.canvasCtx);

            // Handle stages
            this.handleStages(handCenters);

            // Display current stage on canvas
            this.displayStageOnCanvas();
        } else {
            this.updateMessage("No pose detected", "red");
        }

        requestAnimationFrame(() => this.processVideo());
    }

    handleStages(handCenters) {
        if (!handCenters) {
            this.updateMessage("Hands not detected", "red");
            return;
        }

        const { leftHandCenter, rightHandCenter } = handCenters;

        switch (this.currentStage) {
            case this.STAGES.WEBCAM_ON:
                this.updateMessage("Webcam/Video is on. Click 'Start Analysis' to begin.", "blue");
                break;

            case this.STAGES.READY_CHECK:
                this.performReadyCheck(leftHandCenter, rightHandCenter);
                break;

            case this.STAGES.READY:
                this.detectSwingStart(leftHandCenter, rightHandCenter);
                break;

            case this.STAGES.SWING:
                this.calculateSwingAngle(handCenters);
                break;

            case this.STAGES.SHOW_RESULTS:
                // Results have been shown; waiting for next action
                break;

            default:
                break;
        }
    }

    performReadyCheck(leftHandCenter, rightHandCenter) {
        if (!this.initialHandPositions) {
            this.initialHandPositions = { left: leftHandCenter, right: rightHandCenter };
            this.stableStartTime = performance.now();
            this.updateMessage("Please hold still", "blue");
        } else {
            const leftMovement = this.distanceBetweenPoints(leftHandCenter, this.initialHandPositions.left);
            const rightMovement = this.distanceBetweenPoints(rightHandCenter, this.initialHandPositions.right);

            if (leftMovement < this.HAND_STABILITY_THRESHOLD && rightMovement < this.HAND_STABILITY_THRESHOLD) {
                const elapsedTime = performance.now() - this.stableStartTime;
                this.updateMessage(`Hold still... ${Math.max(0, (2 - (elapsedTime / 1000))).toFixed(1)}s`, "blue");

                if (elapsedTime >= 2000) {
                    this.currentStage = this.STAGES.READY;
                    this.updateMessage("Ready! Make your swing!", "green");
                }
            } else {
                this.stableStartTime = performance.now();
                this.initialHandPositions = { left: leftHandCenter, right: rightHandCenter };
                this.updateMessage("Too much movement, please hold still", "red");
            }
        }
    }

    detectSwingStart(leftHandCenter, rightHandCenter) {
        const leftMovement = this.distanceBetweenPoints(leftHandCenter, this.initialHandPositions.left);
        const rightMovement = this.distanceBetweenPoints(rightHandCenter, this.initialHandPositions.right);

        if (leftMovement >= this.SWING_START_THRESHOLD || rightMovement >= this.SWING_START_THRESHOLD) {
            this.currentStage = this.STAGES.SWING;
            this.updateMessage("Swing detected!", "green");
        }
    }

    calculateSwingAngle(handCenters) {
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

        const referenceVector = { x: 0, y: -1 }; // Upward vector
        const dotProduct = shinaiDirection.x * referenceVector.x + shinaiDirection.y * referenceVector.y;
        const angleRadians = Math.acos(dotProduct);
        const angleDegrees = (angleRadians * 180) / Math.PI;

        const cutType = angleDegrees > 45 ? 'Big Cut' : 'Small Cut';

        this.currentStage = this.STAGES.SHOW_RESULTS;
        this.updateMessage(`Swing Angle: ${angleDegrees.toFixed(2)}°<br>Cut Type: ${cutType}<br><br>Click 'Start Analysis' to try again`, "blue");
    }

    distanceBetweenPoints(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    displayStageOnCanvas() {
        this.canvasCtx.font = "20px Arial";
        this.canvasCtx.fillStyle = "yellow";
        let stageText = "";

        switch (this.currentStage) {
            case this.STAGES.IDLE:
                stageText = "Idle";
                break;
            case this.STAGES.WEBCAM_ON:
                stageText = "Webcam/Video On";
                break;
            case this.STAGES.READY_CHECK:
                stageText = "Ready Check";
                break;
            case this.STAGES.READY:
                stageText = "Ready";
                break;
            case this.STAGES.SWING:
                stageText = "Swing";
                break;
            case this.STAGES.SHOW_RESULTS:
                stageText = "Show Results";
                break;
            default:
                stageText = "Unknown Stage";
                break;
        }

        this.canvasCtx.fillText(`Stage: ${stageText}`, 10, 30);
    }
}
