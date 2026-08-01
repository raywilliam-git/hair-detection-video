import {
    FaceLandmarker,
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const webcam =
    document.getElementById("webcam");

const overlay =
    document.getElementById("overlay");

const overlayContext =
    overlay.getContext("2d");

const startButton =
    document.getElementById("startButton");

const stopButton =
    document.getElementById("stopButton");

const toggleVideoButton =
    document.getElementById("toggleVideoButton");

const videoContainer =
    document.getElementById("videoContainer");

const statusDot =
    document.getElementById("statusDot");

const statusText =
    document.getElementById("statusText");

const errorMessage =
    document.getElementById("errorMessage");

const progressBar =
    document.getElementById("progressBar");

const progressText =
    document.getElementById("progressText");


const beepAudio =
    document.getElementById("beepAudio");
    
beepAudio.volume = 1;

const streakSection =
    document.getElementById("streakSection");

const streakTime =
    document.getElementById("streakTime");

const summaryModal =
    document.getElementById("summaryModal");

const summaryDuration =
    document.getElementById("summaryDuration");

const summaryTotal =
    document.getElementById("summaryTotal");

const summaryRate =
    document.getElementById("summaryRate");

const summaryAverageInterval =
    document.getElementById(
        "summaryAverageInterval"
    );

const summaryLongestPeriod =
    document.getElementById(
        "summaryLongestPeriod"
    );

const summaryMostActiveHour =
    document.getElementById(
        "summaryMostActiveHour"
    );

const summaryHourlyChart =
    document.getElementById(
        "summaryHourlyChart"
    );

const downloadReportButton =
    document.getElementById(
        "downloadReportButton"
    );

const closeSummaryButton =
    document.getElementById(
        "closeSummaryButton"
    );

const summarySessionRange =
    document.getElementById(
        "summarySessionRange"
    );


const soundSelect =
    document.getElementById(
        "soundSelect"
    );

const previewSoundButton =
    document.getElementById(
        "previewSoundButton"
    );

let cameraStream = null;
let videoVisible = false;

let faceLandmarker = null;
let handLandmarker = null;

let detectionRunning = false;
let detectionTimer = null;

let currentHairZone = null;


let contactStartTime = null;
let lastBeepTime = null;
let alertActive = false;


/*
 * Session statistics
 */
let sessionStartTime = null;
let sessionEndTime = null;

let reminderEvents = [];

let streakTimer = null;
let lastReminderTime = null;

let lastSessionSummary = null;

/*
 * Configuration
 */
const ANALYSIS_INTERVAL_MS = 250;
const ALERT_DELAY_MS = 2000;
const BEEP_INTERVAL_MS = 1000;

async function initializeModels() {
    statusText.textContent =
        "Loading detection models...";

    const vision =
        await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

    faceLandmarker =
        await FaceLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/" +
                        "mediapipe-models/face_landmarker/" +
                        "face_landmarker/float16/1/" +
                        "face_landmarker.task"
                },

                runningMode: "VIDEO",
                numFaces: 1,

                minFaceDetectionConfidence: 0.5,
                minFacePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5,

                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: false
            }
        );

    handLandmarker =
        await HandLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/" +
                        "mediapipe-models/hand_landmarker/" +
                        "hand_landmarker/float16/1/" +
                        "hand_landmarker.task"
                },

                runningMode: "VIDEO",
                numHands: 2,

                minHandDetectionConfidence: 0.5,
                minHandPresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
            }
        );
}

function startStatisticsSession() {
    sessionStartTime = new Date();
    sessionEndTime = null;

    reminderEvents = [];
    lastReminderTime = null;
    lastSessionSummary = null;

    streakSection.classList.remove("hidden");

    updateStreakDisplay();

    if (streakTimer !== null) {
        window.clearInterval(streakTimer);
    }

    streakTimer = window.setInterval(
        updateStreakDisplay,
        1000
    );
}

function stopStatisticsSession() {
    if (!sessionStartTime) {
        return null;
    }

    sessionEndTime = new Date();

    if (streakTimer !== null) {
        window.clearInterval(streakTimer);
        streakTimer = null;
    }

    streakSection.classList.add("hidden");

    lastSessionSummary =
        calculateSessionSummary();

    return lastSessionSummary;
}

function recordReminderEvent() {
    const eventTime = new Date();

    reminderEvents.push(eventTime);
    lastReminderTime = eventTime;

    updateStreakDisplay();
}

function updateStreakDisplay() {
    if (!sessionStartTime) {
        streakTime.textContent = "00:00";
        return;
    }

    const referenceTime =
        lastReminderTime || sessionStartTime;

    const elapsedMilliseconds =
        Date.now() - referenceTime.getTime();

    streakTime.textContent =
        formatClockDuration(
            elapsedMilliseconds
        );
}

async function startCamera() {
    errorMessage.textContent = "";

    try {
        startButton.disabled = true;

        if (!faceLandmarker || !handLandmarker) {
            await initializeModels();
        }

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    width: {
                        ideal: 640
                    },

                    height: {
                        ideal: 480
                    },

                    frameRate: {
                        ideal: 15,
                        max: 15
                    },

                    facingMode: "user"
                },

                audio: false
            });

        webcam.srcObject = cameraStream;

        await webcam.play();

        resizeOverlay();

        statusDot.classList.add("active");
        statusDot.classList.remove("warning");

        statusText.textContent =
            "Looking for face and hands...";

        stopButton.disabled = false;
        toggleVideoButton.disabled = false;

        videoVisible = true;
        videoContainer.classList.remove("hidden");

        toggleVideoButton.textContent =
            "Hide camera";
            
        startStatisticsSession();

        detectionRunning = true;

        resetContactState();

        scheduleNextDetection();
    } catch (error) {
        console.error(error);

        statusText.textContent = "Error";

        errorMessage.textContent =
            `${error.name || "Error"}: ${
                error.message || String(error)
            }`;

        startButton.disabled = false;
    }
}

function resizeOverlay() {
    if (
        webcam.videoWidth === 0 ||
        webcam.videoHeight === 0
    ) {
        return;
    }

    overlay.width = webcam.videoWidth;
    overlay.height = webcam.videoHeight;
}

function scheduleNextDetection() {
    if (!detectionRunning) {
        return;
    }

    detectionTimer = window.setTimeout(() => {
        if (!detectionRunning) {
            return;
        }

        const currentTime = performance.now();

        analyzeVideoFrame(currentTime);

        scheduleNextDetection();
    }, ANALYSIS_INTERVAL_MS);
}
function analyzeVideoFrame(timestamp) {
    if (
        !faceLandmarker ||
        !handLandmarker ||
        webcam.readyState < 2 ||
        webcam.videoWidth === 0
    ) {
        return;
    }

    if (
        overlay.width !== webcam.videoWidth ||
        overlay.height !== webcam.videoHeight
    ) {
        resizeOverlay();
    }

    const faceResult =
        faceLandmarker.detectForVideo(
            webcam,
            timestamp
        );

    const handResult =
        handLandmarker.detectForVideo(
            webcam,
            timestamp
        );

    processDetection(
        faceResult,
        handResult,
        performance.now()
    );
}

function processDetection(
    faceResult,
    handResult,
    currentTime
) {
    overlayContext.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    currentHairZone = null;

    let faceDetected = false;
    let numberOfHands = 0;
    let handsInHair = 0;

    if (
        faceResult.faceLandmarks &&
        faceResult.faceLandmarks.length > 0
    ) {
        faceDetected = true;

        const faceBounds =
            calculateFaceBounds(
                faceResult.faceLandmarks[0]
            );

        drawFaceRectangle(faceBounds);

        currentHairZone =
            calculateHairZone(faceBounds);

        drawHairZone(currentHairZone);
    }

    if (
        handResult.landmarks &&
        handResult.landmarks.length > 0
    ) {
        numberOfHands =
            handResult.landmarks.length;

        for (
            const landmarks of handResult.landmarks
        ) {
            /*
             * We test several parts of the hand,
             * not only the palm center.
             *
             * This avoids missing contact when the
             * fingers touch the hair while the palm
             * remains outside the orange rectangle.
             */
            const handPoints =
                calculateRelevantHandPoints(landmarks);

            const handInsideHair =
                currentHairZone !== null &&
                handPoints.some(
                    point =>
                        isPointInsideZone(
                            point,
                            currentHairZone
                        )
                );

            if (handInsideHair) {
                handsInHair++;
            }

            const palmCenter =
                calculatePalmCenter(landmarks);

            drawPalmCenter(
                palmCenter,
                handInsideHair
            );
        }
    }

    const anyHandInHair =
        faceDetected && handsInHair > 0;

    updateContactTimer(
        anyHandInHair,
        currentTime
    );

    updateStatus(
        faceDetected,
        numberOfHands,
        handsInHair,
        currentTime
    );
}

function calculateFaceBounds(landmarks) {
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;

    for (const point of landmarks) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);

        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
    }

    return {
        left: minX,
        top: minY,
        right: maxX,
        bottom: maxY,

        width: maxX - minX,
        height: maxY - minY
    };
}

function calculateHairZone(face) {
    return {
        /*
         * Wider than before to cover temples
         * and fingers approaching from the side.
         */
        left: clamp(
            face.left - 0.38 * face.width,
            0,
            1
        ),

        right: clamp(
            face.right + 0.38 * face.width,
            0,
            1
        ),

        top: clamp(
            face.top - 0.55 * face.height,
            0,
            1
        ),

        bottom: clamp(
            face.top + 0.60 * face.height,
            0,
            1
        )
    };
}

function calculatePalmCenter(landmarks) {
    const palmIndexes = [0, 5, 9, 13, 17];

    let sumX = 0;
    let sumY = 0;

    for (const index of palmIndexes) {
        sumX += landmarks[index].x;
        sumY += landmarks[index].y;
    }

    return {
        x: sumX / palmIndexes.length,
        y: sumY / palmIndexes.length
    };
}

function calculateRelevantHandPoints(landmarks) {
    const palmCenter =
        calculatePalmCenter(landmarks);

    return [
        palmCenter,

        /*
         * Fingertips:
         * thumb, index, middle, ring, little finger.
         */
        landmarks[4],
        landmarks[8],
        landmarks[12],
        landmarks[16],
        landmarks[20],

        /*
         * Middle finger joints, useful when fingers
         * are curled into the hair.
         */
        landmarks[6],
        landmarks[10],
        landmarks[14],
        landmarks[18]
    ];
}

function isPointInsideZone(point, zone) {
    return (
        point.x >= zone.left &&
        point.x <= zone.right &&
        point.y >= zone.top &&
        point.y <= zone.bottom
    );
}

function updateContactTimer(
    anyHandInHair,
    currentTime
) {
    if (!anyHandInHair) {
        resetContactState();
        return;
    }

    if (contactStartTime === null) {
        contactStartTime = currentTime;
        lastBeepTime = null;
        alertActive = false;
    }

    const elapsed =
        currentTime - contactStartTime;

    if (
        elapsed >= ALERT_DELAY_MS &&
        !alertActive
    ) {
        alertActive = true;
        lastBeepTime = currentTime;

        /*
        * One statistical event per continuous
        * hand-in-hair episode.
        */
        recordReminderEvent();

        playBeep();
    }
    if (
        alertActive &&
        currentTime - lastBeepTime >=
            BEEP_INTERVAL_MS
    ) {
        lastBeepTime = currentTime;

        playBeep();
    }

    updateProgress(elapsed);
}

function updateProgress(elapsedMilliseconds) {
    const cappedElapsed =
        Math.min(
            elapsedMilliseconds,
            ALERT_DELAY_MS
        );

    const percentage =
        cappedElapsed /
        ALERT_DELAY_MS *
        100;

    progressBar.style.width =
        `${percentage}%`;

    progressText.textContent =
        `${(
            elapsedMilliseconds / 1000
        ).toFixed(1)} s / 2.0 s`;

    progressBar.classList.toggle(
        "alert",
        elapsedMilliseconds >= ALERT_DELAY_MS
    );
}

function resetContactState() {
    contactStartTime = null;
    lastBeepTime = null;
    alertActive = false;

    progressBar.style.width = "0%";
    progressBar.classList.remove("alert");

    progressText.textContent =
        "0.0 s / 2.0 s";

    statusDot.classList.remove("warning");
}

async function playBeep() {
    try {
        /*
         * Restart the sound from the beginning,
         * even if it was already playing.
         */
        beepAudio.pause();
        beepAudio.currentTime = 0;

        await beepAudio.play();
    } catch (error) {
        console.warn(
            "Unable to play reminder sound:",
            error
        );
    }
}
function drawFaceRectangle(face) {
    const x =
        face.left * overlay.width;

    const y =
        face.top * overlay.height;

    const width =
        face.width * overlay.width;

    const height =
        face.height * overlay.height;

    overlayContext.strokeStyle = "#20c060";
    overlayContext.lineWidth = 3;

    overlayContext.strokeRect(
        x,
        y,
        width,
        height
    );
}

function drawHairZone(zone) {
    const x =
        zone.left * overlay.width;

    const y =
        zone.top * overlay.height;

    const width =
        (zone.right - zone.left) *
        overlay.width;

    const height =
        (zone.bottom - zone.top) *
        overlay.height;

    overlayContext.strokeStyle = "#ff9800";
    overlayContext.lineWidth = 4;

    overlayContext.setLineDash([10, 7]);

    overlayContext.strokeRect(
        x,
        y,
        width,
        height
    );

    overlayContext.setLineDash([]);
}

function drawPalmCenter(
    palmCenter,
    insideHair
) {
    const x =
        palmCenter.x * overlay.width;

    const y =
        palmCenter.y * overlay.height;

    overlayContext.beginPath();

    overlayContext.fillStyle =
        insideHair
            ? "#ef233c"
            : "#168aad";

    overlayContext.arc(
        x,
        y,
        insideHair ? 15 : 11,
        0,
        Math.PI * 2
    );

    overlayContext.fill();

    overlayContext.strokeStyle = "white";
    overlayContext.lineWidth = 3;
    overlayContext.stroke();
}

function updateStatus(
    faceDetected,
    numberOfHands,
    handsInHair,
    currentTime
) {
    statusDot.classList.toggle(
        "warning",
        alertActive
    );

    if (!faceDetected) {
        statusText.textContent =
            "No face detected";

        return;
    }

    if (handsInHair > 0) {
        const elapsed =
            contactStartTime === null
                ? 0
                : currentTime -
                  contactStartTime;

        statusText.textContent =
            alertActive
                ? `${handsInHair} hand(s) in hair zone — alert active`
                : `${handsInHair} hand(s) in hair zone — ${(
                    elapsed / 1000
                  ).toFixed(1)} s`;

        return;
    }

    if (numberOfHands === 0) {
        statusText.textContent =
            "Face detected — no hands detected";

        return;
    }

    statusText.textContent =
        `Face detected — ${numberOfHands} hand(s) detected`;
}

function clamp(
    value,
    minimum,
    maximum
) {
    return Math.min(
        Math.max(value, minimum),
        maximum
    );
}

function toggleVideo() {
    videoVisible = !videoVisible;

    videoContainer.classList.toggle(
        "hidden",
        !videoVisible
    );

    toggleVideoButton.textContent =
        videoVisible
            ? "Hide camera"
            : "Show camera";
}



function calculateSessionSummary() {
    const start = sessionStartTime;
    const end = sessionEndTime || new Date();

    const durationMilliseconds =
        Math.max(
            0,
            end.getTime() - start.getTime()
        );

    const totalReminders =
        reminderEvents.length;

    const durationHours =
        durationMilliseconds / 3600000;

    const remindersPerHour =
        durationHours > 0
            ? totalReminders / durationHours
            : 0;

    const intervals = [];

    for (
        let index = 1;
        index < reminderEvents.length;
        index++
    ) {
        intervals.push(
            reminderEvents[index].getTime() -
            reminderEvents[index - 1].getTime()
        );
    }

    const averageIntervalMilliseconds =
        intervals.length > 0
            ? intervals.reduce(
                (sum, interval) =>
                    sum + interval,
                0
            ) / intervals.length
            : null;

    const reminderFreePeriods =
        calculateReminderFreePeriods(
            start,
            end,
            reminderEvents
        );

    const longestPeriod =
        reminderFreePeriods.reduce(
            (longest, period) =>
                !longest ||
                period.durationMilliseconds >
                    longest.durationMilliseconds
                    ? period
                    : longest,
            null
        );

    const hourlyCounts =
        calculateHourlyCounts(reminderEvents);

    const mostActiveHour =
        findMostActiveHour(hourlyCounts);

    return {
        start,
        end,
        durationMilliseconds,
        totalReminders,
        remindersPerHour,
        averageIntervalMilliseconds,
        longestPeriod,
        hourlyCounts,
        mostActiveHour,

        firstReminder:
            reminderEvents.length > 0
                ? reminderEvents[0]
                : null,

        lastReminder:
            reminderEvents.length > 0
                ? reminderEvents[
                    reminderEvents.length - 1
                  ]
                : null,

        events: [...reminderEvents]
    };
}

function calculateReminderFreePeriods(
    start,
    end,
    events
) {
    if (events.length === 0) {
        return [
            {
                start,
                end,
                durationMilliseconds:
                    end.getTime() -
                    start.getTime()
            }
        ];
    }

    const periods = [];

    periods.push({
        start,
        end: events[0],

        durationMilliseconds:
            events[0].getTime() -
            start.getTime()
    });

    for (
        let index = 1;
        index < events.length;
        index++
    ) {
        periods.push({
            start: events[index - 1],
            end: events[index],

            durationMilliseconds:
                events[index].getTime() -
                events[index - 1].getTime()
        });
    }

    periods.push({
        start: events[events.length - 1],
        end,

        durationMilliseconds:
            end.getTime() -
            events[events.length - 1].getTime()
    });

    return periods;
}

function calculateHourlyCounts(events) {
    const counts = new Map();

    for (const event of events) {
        const hourStart =
            new Date(event);

        hourStart.setMinutes(0, 0, 0);

        const key =
            hourStart.toISOString();

        if (!counts.has(key)) {
            counts.set(key, {
                start: hourStart,
                count: 0
            });
        }

        counts.get(key).count++;
    }

    return Array.from(counts.values())
        .sort(
            (first, second) =>
                first.start -
                second.start
        );
}

function findMostActiveHour(hourlyCounts) {
    if (hourlyCounts.length === 0) {
        return null;
    }

    return hourlyCounts.reduce(
        (mostActive, current) =>
            current.count > mostActive.count
                ? current
                : mostActive
    );
}

function stopCamera() {
    const summary =
        stopStatisticsSession();

    detectionRunning = false;

    if (detectionTimer !== null) {
        window.clearTimeout(detectionTimer);
        detectionTimer = null;
    }

    currentHairZone = null;

    resetContactState();

    if (cameraStream) {
        for (
            const track of
            cameraStream.getTracks()
        ) {
            track.stop();
        }
    }

    webcam.srcObject = null;
    cameraStream = null;

    overlayContext.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    videoVisible = false;
    videoContainer.classList.add("hidden");

    statusDot.classList.remove(
        "active",
        "warning"
    );

    statusText.textContent =
        "Camera inactive";

    startButton.disabled = false;
    stopButton.disabled = true;
    toggleVideoButton.disabled = true;

    toggleVideoButton.textContent =
        "Show camera";
    if (summary) {
        showSessionSummary(summary);
    }
}

startButton.addEventListener(
    "click",
    async () => {
        try {
            beepAudio.currentTime = 0;
            await beepAudio.play();

            beepAudio.pause();
            beepAudio.currentTime = 0;
        } catch (audioError) {
            console.warn(
                "Audio could not be enabled:",
                audioError
            );
        }
        await startCamera();
    }
);

soundSelect.addEventListener(
    "change",
    () => {

        beepAudio.src =
            `assets/sounds/${soundSelect.value}`;

    }
);

previewSoundButton.addEventListener(
    "click",
    async () => {

        try {

            beepAudio.pause();

            beepAudio.currentTime = 0;

            await beepAudio.play();

        }

        catch(error) {

            console.warn(error);

        }

    }
);

toggleVideoButton.addEventListener(
    "click",
    toggleVideo
);

stopButton.addEventListener(
    "click",
    stopCamera
);

window.addEventListener(
    "resize",
    resizeOverlay
);

function showSessionSummary(summary) {
    summaryDuration.textContent =
        formatLongDuration(
            summary.durationMilliseconds
        );
    summarySessionRange.textContent =
    `${formatTime(summary.start)}–${formatTime(summary.end)}`;

    summaryTotal.textContent =
        summary.totalReminders.toString();

    summaryRate.textContent =
    summary.durationMilliseconds <
        5 * 60 * 1000
        ? "Available after 5 min"
        : summary.remindersPerHour.toFixed(1);


    summaryAverageInterval.textContent =
        summary.averageIntervalMilliseconds === null
            ? "Not enough events"
            : formatLongDuration(
                summary.averageIntervalMilliseconds
            );

    summaryLongestPeriod.textContent =
    summary.longestPeriod
        ? formatLongDuration(
            summary.longestPeriod
                .durationMilliseconds
        )
        : "—";

    summaryMostActiveHour.textContent =
        summary.mostActiveHour
            ? formatHourRange(
                summary.mostActiveHour.start
            )
            : "No reminders";
    renderHourlyChart(summary.hourlyCounts);
    summaryModal.classList.remove("hidden");
}


function buildTextReport(summary) {
    const lines = [];

    lines.push(
        "HAIR TOUCH REMINDER — SESSION REPORT"
    );

    lines.push("");

    lines.push(
        `Date: ${formatDate(summary.start)}`
    );

    lines.push(
        `Session start: ${formatTime(summary.start)}`
    );

    lines.push(
        `Session end: ${formatTime(summary.end)}`
    );

    lines.push(
        `Session duration: ${formatLongDuration(
            summary.durationMilliseconds
        )}`
    );

    lines.push("");
    lines.push("SUMMARY");
    lines.push("");

    lines.push(
        `Total reminders: ${summary.totalReminders}`
    );

    lines.push(
        `Reminders per hour: ${
            summary.durationMilliseconds <
                5 * 60 * 1000
                ? "Not meaningful for sessions under 5 minutes"
                : summary.remindersPerHour.toFixed(1)
        }`
    );

    lines.push(
        `Average time between reminders: ${
            summary.averageIntervalMilliseconds === null
                ? "Not enough events"
                : formatLongDuration(
                    summary
                        .averageIntervalMilliseconds
                )
        }`
    );

    if (summary.longestPeriod) {
        lines.push(
            `Longest reminder-free period: ` +
            `${formatLongDuration(
                summary.longestPeriod
                    .durationMilliseconds
            )}`
        );

        lines.push(
            `Longest period range: ` +
            `${formatTime(
                summary.longestPeriod.start
            )}–${formatTime(
                summary.longestPeriod.end
            )}`
        );
    }

    lines.push(
        `Most active hour: ${
            summary.mostActiveHour
                ? formatHourRange(
                    summary.mostActiveHour.start
                )
                : "No reminders"
        }`
    );

    lines.push(
        `First reminder: ${
            summary.firstReminder
                ? formatTime(summary.firstReminder)
                : "No reminders"
        }`
    );

    lines.push(
        `Last reminder: ${
            summary.lastReminder
                ? formatTime(summary.lastReminder)
                : "No reminders"
        }`
    );

    lines.push("");
    lines.push("EVENTS");
    lines.push("");

    if (summary.events.length === 0) {
        lines.push("No reminders during this session.");
    } else {
        summary.events.forEach(
            (event, index) => {
                lines.push(
                    `${index + 1}. ${formatTime(event)}`
                );
            }
        );
    }

    lines.push("");
    lines.push("EVENTS BY HOUR");
    lines.push("");

    if (summary.hourlyCounts.length === 0) {
        lines.push("No reminders during this session.");
    } else {
        const maximumCount = Math.max(
            ...summary.hourlyCounts.map(
                item => item.count
            )
        );

        for (
            const item of summary.hourlyCounts
        ) {
            const barLength =
                Math.max(
                    1,
                    Math.round(
                        item.count /
                        maximumCount *
                        20
                    )
                );

            const bar = "█".repeat(barLength);

            lines.push(
                `${formatHourRange(item.start)}  ` +
                `${bar} ${item.count}`
            );
        }
    }

    return lines.join("\n");
}

function downloadTextReport() {
    if (!lastSessionSummary) {
        return;
    }

    const report =
        buildTextReport(lastSessionSummary);

    const blob =
        new Blob(
            [report],
            {
                type:
                    "text/plain;charset=utf-8"
            }
        );

    const downloadUrl =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = downloadUrl;

    link.download =
        `hair-touch-session-${
            formatFileDate(
                lastSessionSummary.start
            )
        }.txt`;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(downloadUrl);
}

function formatClockDuration(milliseconds) {
    const totalSeconds =
        Math.max(
            0,
            Math.floor(milliseconds / 1000)
        );

    const hours =
        Math.floor(totalSeconds / 3600);

    const minutes =
        Math.floor(
            (totalSeconds % 3600) / 60
        );

    const seconds =
        totalSeconds % 60;

    if (hours > 0) {
        return [
            hours,
            minutes,
            seconds
        ]
            .map(value =>
                String(value).padStart(2, "0")
            )
            .join(":");
    }

    return [
        minutes,
        seconds
    ]
        .map(value =>
            String(value).padStart(2, "0")
        )
        .join(":");
}

function formatLongDuration(milliseconds) {
    const totalSeconds =
        Math.max(
            0,
            Math.round(milliseconds / 1000)
        );

    const hours =
        Math.floor(totalSeconds / 3600);

    const minutes =
        Math.floor(
            (totalSeconds % 3600) / 60
        );

    const seconds =
        totalSeconds % 60;

    const parts = [];

    if (hours > 0) {
        parts.push(
            `${hours} h`
        );
    }

    if (minutes > 0 || hours > 0) {
        parts.push(
            `${minutes} min`
        );
    }

    parts.push(
        `${seconds} s`
    );

    return parts.join(" ");
}

function formatTime(date) {
    return date.toLocaleTimeString(
        [],
        {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }
    );
}

function formatDate(date) {
    return date.toLocaleDateString(
        "en-GB",
        {
            day: "numeric",
            month: "long",
            year: "numeric"
        }
    );
}

function formatHourRange(date) {
    const start =
        new Date(date);

    const end =
        new Date(start);

    end.setHours(
        end.getHours() + 1
    );

    return (
        `${start.toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit"
            }
        )}–` +
        `${end.toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit"
            }
        )}`
    );
}

function formatFileDate(date) {
    const year =
        date.getFullYear();

    const month =
        String(
            date.getMonth() + 1
        ).padStart(2, "0");

    const day =
        String(
            date.getDate()
        ).padStart(2, "0");

    const hours =
        String(
            date.getHours()
        ).padStart(2, "0");

    const minutes =
        String(
            date.getMinutes()
        ).padStart(2, "0");

    return (
        `${year}-${month}-${day}` +
        `_${hours}-${minutes}`
    );
}

downloadReportButton.addEventListener(
    "click",
    downloadTextReport
);

closeSummaryButton.addEventListener(
    "click",
    () => {
        summaryModal.classList.add("hidden");
    }
);


function renderHourlyChart(hourlyCounts) {
    summaryHourlyChart.innerHTML = "";

    if (hourlyCounts.length === 0) {
        const emptyMessage =
            document.createElement("p");

        emptyMessage.className =
            "chart-empty-message";

        emptyMessage.textContent =
            "No reminders during this session.";

        summaryHourlyChart.appendChild(
            emptyMessage
        );

        return;
    }

    const maximumCount = Math.max(
        ...hourlyCounts.map(
            item => item.count
        )
    );

    for (const item of hourlyCounts) {
        const row =
            document.createElement("div");

        row.className = "hourly-chart-row";

        const label =
            document.createElement("span");

        label.className =
            "hourly-chart-label";

        label.textContent =
            formatHourRange(item.start);

        const track =
            document.createElement("div");

        track.className =
            "hourly-chart-track";

        const bar =
            document.createElement("div");

        bar.className =
            "hourly-chart-bar";

        bar.style.width =
            `${(
                item.count /
                maximumCount *
                100
            ).toFixed(1)}%`;

        const value =
            document.createElement("strong");

        value.className =
            "hourly-chart-value";

        value.textContent =
            item.count.toString();

        track.appendChild(bar);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);

        summaryHourlyChart.appendChild(row);
    }
}