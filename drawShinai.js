// drawShinai.js

export function drawShinai(landmarks, canvasElement, canvasCtx) {
    const landmarksIndices = {
        LEFT_THUMB: 21,
        LEFT_INDEX: 19,
        LEFT_PINKY: 17,
        RIGHT_THUMB: 22,
        RIGHT_INDEX: 20,
        RIGHT_PINKY: 18,
    };

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
        return null;
    }

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
    canvasCtx.closePath();
    canvasCtx.strokeStyle = 'red';
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();

    // Draw right hand grip
    canvasCtx.beginPath();
    canvasCtx.moveTo(rightHandPoints.thumb.x, rightHandPoints.thumb.y);
    canvasCtx.lineTo(rightHandPoints.index.x, rightHandPoints.index.y);
    canvasCtx.lineTo(rightHandPoints.pinky.x, rightHandPoints.pinky.y);
    canvasCtx.closePath();
    canvasCtx.strokeStyle = 'blue';
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();

    // Draw connecting line between hands (shinai)
    const leftHandCenter = {
        x: (leftHandPoints.thumb.x + leftHandPoints.index.x + leftHandPoints.pinky.x) / 3,
        y: (leftHandPoints.thumb.y + leftHandPoints.index.y + leftHandPoints.pinky.y) / 3
    };
    const rightHandCenter = {
        x: (rightHandPoints.thumb.x + rightHandPoints.index.x + rightHandPoints.pinky.x) / 3,
        y: (rightHandPoints.thumb.y + rightHandPoints.index.y + rightHandPoints.pinky.y) / 3
    };

    canvasCtx.beginPath();
    canvasCtx.moveTo(leftHandCenter.x, leftHandCenter.y);
    canvasCtx.lineTo(rightHandCenter.x, rightHandCenter.y);
    canvasCtx.strokeStyle = 'purple';
    canvasCtx.lineWidth = 4;
    canvasCtx.stroke();

    return { leftHandCenter, rightHandCenter };
}
