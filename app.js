import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const landmarksIndices = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32
};


// Variables to manage pose detection and webcam state
let poseLandmarker = undefined;
let runningMode = "VIDEO";
let enableWebcamButton;
let webcamRunning = false;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// Initialize the pose landmarker
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

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

async function enableCam() {
  console.log('Enabling webcam...');
  if (!poseLandmarker) {
    console.log("Wait! poseLandmarker not loaded yet.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE WEBCAM";
    console.log('Webcam disabled.');
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE WEBCAM";
    console.log('Webcam enabled.');
  }

  const constraints = {
    video: true
  };

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

async function predictWebcam() {
  console.log('Starting webcam prediction...');
  const drawingUtils = new DrawingUtils(canvasCtx);

  async function onFrame() {
    if (!webcamRunning) {
      console.log('Webcam not running, exiting onFrame.');
      return;
    }

    const results = await poseLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks && results.landmarks[0]) {
      // Access the first set of landmarks detected
      const landmarks = results.landmarks[0];
      calculateSwingAngle(landmarks);
      
      // Draw pose landmarks and connectors
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
      drawingUtils.drawLandmarks(landmarks, { radius: 5 });
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);
    } else {
      console.log("No landmarks detected");
    }

    requestAnimationFrame(onFrame);
  }

  onFrame();
}

function calculateSwingAngle(landmarks) {
  console.log('Calculating swing angle...');

  // Access the wrist landmarks using landmarksIndices
  const leftWrist = landmarks[landmarksIndices.LEFT_WRIST];
  const rightWrist = landmarks[landmarksIndices.RIGHT_WRIST];

  if (!leftWrist || !rightWrist) {
    console.log("Could not find wrist landmarks");
    return;
  }

  // Calculate the vector representing the shinai (from right wrist to left wrist)
  const shinaiVector = {
    x: leftWrist.x - rightWrist.x,
    y: leftWrist.y - rightWrist.y,
    z: leftWrist.z - rightWrist.z
  };

  // Normalize the vector to find direction
  const magnitude = Math.sqrt(
    shinaiVector.x ** 2 + shinaiVector.y ** 2 + shinaiVector.z ** 2
  );
  const shinaiDirection = {
    x: shinaiVector.x / magnitude,
    y: shinaiVector.y / magnitude,
    z: shinaiVector.z / magnitude
  };

  // Define a vertical reference vector
  const referenceVector = { x: 0, y: -1, z: 0 }; // Negative y-axis

  // Calculate the dot product to find the angle between shinai direction and vertical
  const dotProduct =
    shinaiDirection.x * referenceVector.x +
    shinaiDirection.y * referenceVector.y +
    shinaiDirection.z * referenceVector.z;

  // Calculate the angle in radians and convert to degrees
  const angleRadians = Math.acos(dotProduct);
  const angleDegrees = (angleRadians * 180) / Math.PI;

  // Classify the cut based on the angle
  let cutType;
  if (angleDegrees > 45) {
    cutType = 'Big Cut';
  } else {
    cutType = 'Small Cut';
  }

  // Display the calculated angle and cut type on the canvas
  canvasCtx.font = "16px Arial";
  canvasCtx.fillStyle = "red";
  canvasCtx.fillText(`Swing Angle: ${angleDegrees.toFixed(2)}°`, 10, 20);
  canvasCtx.fillText(`Cut Type: ${cutType}`, 10, 40);

  // Draw the approximated shinai line on the canvas
  drawShinai(leftWrist, shinaiDirection);
}

function drawShinai(leftWrist, shinaiDirection) {
  const shinaiLength = 0.5; // Adjust length as needed

  // Calculate the endpoint of the shinai line based on its direction
  const shinaiEndPoint = {
    x: leftWrist.x + shinaiDirection.x * shinaiLength,
    y: leftWrist.y + shinaiDirection.y * shinaiLength
  };

  // Scale coordinates to the canvas size
  const startX = leftWrist.x * canvasElement.width;
  const startY = leftWrist.y * canvasElement.height;
  const endX = shinaiEndPoint.x * canvasElement.width;
  const endY = shinaiEndPoint.y * canvasElement.height;

  // Draw the shinai line
  canvasCtx.beginPath();
  canvasCtx.moveTo(startX, startY);
  canvasCtx.lineTo(endX, endY);
  canvasCtx.strokeStyle = 'blue';
  canvasCtx.lineWidth = 4;
  canvasCtx.stroke();
}
