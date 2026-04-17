const state = {
    sessionId: null,
    imageWidth: 1024,
    imageHeight: 1024,
    featureWidth: 256,
    featureHeight: 256,
    canvasImage: null,
    pendingHandle: null,
    pairs: [],
};

const els = {
    healthBadge: document.getElementById("healthBadge"),
    statusBar: document.getElementById("statusBar"),
    sourceModeInputs: Array.from(document.querySelectorAll("input[name='sourceMode']")),

    generateForm: document.getElementById("generateForm"),
    seedInput: document.getElementById("seedInput"),
    generationStepsInput: document.getElementById("generationStepsInput"),
    generateBtn: document.getElementById("generateBtn"),

    invertForm: document.getElementById("invertForm"),
    imageInput: document.getElementById("imageInput"),
    inversionStepsInput: document.getElementById("inversionStepsInput"),
    inversionLrInput: document.getElementById("inversionLrInput"),
    inversionNoiseRegInput: document.getElementById("inversionNoiseRegInput"),
    inversionFullresInput: document.getElementById("inversionFullresInput"),
    inversionLowresInput: document.getElementById("inversionLowresInput"),
    inversionGradInput: document.getElementById("inversionGradInput"),
    invertBtn: document.getElementById("invertBtn"),

    pickerCanvas: document.getElementById("pickerCanvas"),
    undoPairBtn: document.getElementById("undoPairBtn"),
    clearPairsBtn: document.getElementById("clearPairsBtn"),
    pairCount: document.getElementById("pairCount"),
    pairList: document.getElementById("pairList"),

    dragForm: document.getElementById("dragForm"),
    dragStepsInput: document.getElementById("dragStepsInput"),
    dragBtn: document.getElementById("dragBtn"),

    sourceCard: document.getElementById("sourceCard"),
    sourceImage: document.getElementById("sourceImage"),
    initialImage: document.getElementById("initialImage"),
    initialFeatureMap: document.getElementById("initialFeatureMap"),
    initialFeatureTensorLink: document.getElementById("initialFeatureTensorLink"),
    outputImage: document.getElementById("outputImage"),
    outputFeatureMap: document.getElementById("outputFeatureMap"),
    outputFeatureTensorLink: document.getElementById("outputFeatureTensorLink"),
};

const canvasCtx = els.pickerCanvas.getContext("2d");

function withBust(url) {
    if (!url) {
        return "";
    }
    const stamp = Date.now();
    if (url.includes("?")) {
        return `${url}&t=${stamp}`;
    }
    return `${url}?t=${stamp}`;
}

function setStatus(message, kind = "info") {
    els.statusBar.textContent = message;
    els.statusBar.classList.remove("error", "info");
    if (kind === "error") {
        els.statusBar.classList.add("error");
    } else {
        els.statusBar.classList.add("info");
    }
}

function setHealth(message, healthy) {
    els.healthBadge.textContent = message;
    if (healthy) {
        els.healthBadge.style.borderColor = "rgba(30, 143, 122, 0.35)";
        els.healthBadge.style.color = "#1d4e44";
    } else {
        els.healthBadge.style.borderColor = "rgba(189, 71, 42, 0.4)";
        els.healthBadge.style.color = "#7c2f19";
    }
}

function setBusy(isBusy) {
    els.generateBtn.disabled = isBusy;
    els.invertBtn.disabled = isBusy;
    els.dragBtn.disabled = isBusy;
}

async function requestJson(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    const body = await response.json().catch(() => ({ ok: false, error: "Invalid server response." }));
    if (!response.ok || !body.ok) {
        throw new Error(body.error || `Request failed: ${response.status}`);
    }

    return body.data;
}

async function requestForm(url, formData) {
    const response = await fetch(url, {
        method: "POST",
        body: formData,
    });

    const body = await response.json().catch(() => ({ ok: false, error: "Invalid server response." }));
    if (!response.ok || !body.ok) {
        throw new Error(body.error || `Request failed: ${response.status}`);
    }

    return body.data;
}

function parseOptionalInt(rawValue) {
    if (rawValue === "") {
        return null;
    }
    const parsed = Number.parseInt(rawValue, 10);
    if (Number.isNaN(parsed)) {
        return null;
    }
    return parsed;
}

function selectedMode() {
    const active = els.sourceModeInputs.find((input) => input.checked);
    return active ? active.value : "generate";
}

function refreshModeVisibility() {
    const mode = selectedMode();
    if (mode === "generate") {
        els.generateForm.classList.remove("hidden");
        els.invertForm.classList.add("hidden");
    } else {
        els.generateForm.classList.add("hidden");
        els.invertForm.classList.remove("hidden");
    }
}

function clearPairState() {
    state.pendingHandle = null;
    state.pairs = [];
    renderPairList();
    drawCanvas();
}

function renderPairList() {
    els.pairList.innerHTML = "";

    state.pairs.forEach((pair, idx) => {
        const item = document.createElement("li");
        item.textContent = `Pair ${idx + 1}: H(${pair.handle.x}, ${pair.handle.y}) -> T(${pair.target.x}, ${pair.target.y})`;
        els.pairList.appendChild(item);
    });

    if (state.pendingHandle) {
        const item = document.createElement("li");
        item.textContent = `Pending handle: (${state.pendingHandle.x}, ${state.pendingHandle.y})`;
        els.pairList.appendChild(item);
    }

    els.pairCount.textContent = `${state.pairs.length} pair${state.pairs.length === 1 ? "" : "s"}`;
}

function drawPoint(x, y, color, size = 6) {
    canvasCtx.beginPath();
    canvasCtx.arc(x, y, size, 0, Math.PI * 2);
    canvasCtx.fillStyle = color;
    canvasCtx.fill();
}

function drawCanvas() {
    const canvas = els.pickerCanvas;

    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.canvasImage) {
        canvasCtx.fillStyle = "#f0ece6";
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        canvasCtx.fillStyle = "#5f6f82";
        canvasCtx.font = "20px Space Grotesk";
        canvasCtx.fillText("Create a session to start selecting points", 40, 60);
        return;
    }

    canvasCtx.drawImage(state.canvasImage, 0, 0, canvas.width, canvas.height);

    state.pairs.forEach((pair, idx) => {
        canvasCtx.strokeStyle = "rgba(12, 50, 81, 0.85)";
        canvasCtx.lineWidth = 2;
        canvasCtx.beginPath();
        canvasCtx.moveTo(pair.handle.x, pair.handle.y);
        canvasCtx.lineTo(pair.target.x, pair.target.y);
        canvasCtx.stroke();

        drawPoint(pair.handle.x, pair.handle.y, "#e56b2f", 6);
        drawPoint(pair.target.x, pair.target.y, "#1e8f7a", 6);

        canvasCtx.fillStyle = "#12283d";
        canvasCtx.font = "15px IBM Plex Mono";
        canvasCtx.fillText(`${idx + 1}`, pair.handle.x + 8, pair.handle.y - 8);
    });

    if (state.pendingHandle) {
        canvasCtx.setLineDash([8, 8]);
        canvasCtx.strokeStyle = "#e56b2f";
        canvasCtx.lineWidth = 2;
        canvasCtx.beginPath();
        canvasCtx.arc(state.pendingHandle.x, state.pendingHandle.y, 10, 0, Math.PI * 2);
        canvasCtx.stroke();
        canvasCtx.setLineDash([]);
    }
}

function loadImageIntoCanvas(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            state.canvasImage = image;
            els.pickerCanvas.width = image.naturalWidth;
            els.pickerCanvas.height = image.naturalHeight;
            state.imageWidth = image.naturalWidth;
            state.imageHeight = image.naturalHeight;
            drawCanvas();
            resolve();
        };
        image.onerror = () => reject(new Error("Could not load image for canvas."));
        image.src = withBust(url);
    });
}

function canvasClickToPoint(event) {
    const rect = els.pickerCanvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * els.pickerCanvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * els.pickerCanvas.height;
    return {
        x: Math.max(0, Math.min(Math.round(x), els.pickerCanvas.width - 1)),
        y: Math.max(0, Math.min(Math.round(y), els.pickerCanvas.height - 1)),
    };
}

function applyOutputLinks(data) {
    if (data.output_image_url) {
        els.outputImage.src = withBust(data.output_image_url);
    }
    if (data.output_feature_map_url) {
        els.outputFeatureMap.src = withBust(data.output_feature_map_url);
    }
    if (data.output_feature_tensor_url) {
        els.outputFeatureTensorLink.href = data.output_feature_tensor_url;
    }
}

async function applySessionData(data) {
    state.sessionId = data.session_id;
    state.featureHeight = data.feature_size.height;
    state.featureWidth = data.feature_size.width;

    clearPairState();

    if (data.source_image_url) {
        els.sourceCard.classList.remove("hidden");
        els.sourceImage.src = withBust(data.source_image_url);
    } else {
        els.sourceCard.classList.add("hidden");
        els.sourceImage.removeAttribute("src");
    }

    els.initialImage.src = withBust(data.initial_image_url);
    els.initialFeatureMap.src = withBust(data.initial_feature_map_url);
    els.initialFeatureTensorLink.href = data.initial_feature_tensor_url;

    els.outputImage.removeAttribute("src");
    els.outputFeatureMap.removeAttribute("src");
    els.outputFeatureTensorLink.href = "#";

    await loadImageIntoCanvas(data.initial_image_url);

    setStatus(
        `Session ${data.session_id} ready on ${data.device}. Image ${data.image_size.width}x${data.image_size.height}, feature map ${data.feature_size.width}x${data.feature_size.height}.`,
        "info"
    );
}

async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const body = await response.json();
        if (!response.ok || !body.ok) {
            throw new Error(body.error || "Backend unavailable");
        }
        setHealth(`Backend ready on ${body.data.device.toUpperCase()}`, true);
    } catch (error) {
        setHealth("Backend offline", false);
    }
}

els.sourceModeInputs.forEach((input) => {
    input.addEventListener("change", () => {
        refreshModeVisibility();
    });
});

els.generateForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const seed = parseOptionalInt(els.seedInput.value.trim());
    const generationSteps = Math.max(1, Number.parseInt(els.generationStepsInput.value, 10) || 1);

    setBusy(true);
    setStatus("Generating initial image and feature map...", "info");
    try {
        const data = await requestJson("/api/generate", {
            seed,
            generation_steps: generationSteps,
        });
        await applySessionData(data);
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
});

els.invertForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const imageFile = els.imageInput.files[0];
    if (!imageFile) {
        setStatus("Choose an image file before inversion.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("inversion_steps", String(Math.max(1, Number.parseInt(els.inversionStepsInput.value, 10) || 900)));
    formData.append("inversion_lr", String(Number.parseFloat(els.inversionLrInput.value) || 0.05));
    formData.append("inversion_noise_reg", String(Number.parseFloat(els.inversionNoiseRegInput.value) || 30000));
    formData.append("inversion_fullres_weight", String(Number.parseFloat(els.inversionFullresInput.value) || 1.0));
    formData.append("inversion_lowres_weight", String(Number.parseFloat(els.inversionLowresInput.value) || 0.25));
    formData.append("inversion_gradient_weight", String(Number.parseFloat(els.inversionGradInput.value) || 0.35));

    setBusy(true);
    setStatus("Running GAN inversion. This can take a while...", "info");
    try {
        const data = await requestForm("/api/invert", formData);
        await applySessionData(data);
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
});

els.pickerCanvas.addEventListener("click", (event) => {
    if (!state.sessionId || !state.canvasImage) {
        setStatus("Create a session first, then click points.", "error");
        return;
    }

    const point = canvasClickToPoint(event);

    if (state.pendingHandle === null) {
        state.pendingHandle = point;
    } else {
        state.pairs.push({
            handle: state.pendingHandle,
            target: point,
        });
        state.pendingHandle = null;
    }

    renderPairList();
    drawCanvas();
});

els.undoPairBtn.addEventListener("click", () => {
    if (state.pendingHandle) {
        state.pendingHandle = null;
    } else {
        state.pairs.pop();
    }
    renderPairList();
    drawCanvas();
});

els.clearPairsBtn.addEventListener("click", () => {
    clearPairState();
});

els.dragForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!state.sessionId) {
        setStatus("No active session. Generate or invert first.", "error");
        return;
    }

    if (state.pairs.length === 0) {
        setStatus("Add at least one handle-target pair.", "error");
        return;
    }

    const dragSteps = Math.max(1, Number.parseInt(els.dragStepsInput.value, 10) || 50);

    setBusy(true);
    setStatus("Running drag optimization...", "info");

    try {
        const data = await requestJson("/api/drag", {
            session_id: state.sessionId,
            pairs: state.pairs,
            drag_steps: dragSteps,
        });

        applyOutputLinks(data);

        if (Array.isArray(data.updated_handles_image) && data.updated_handles_image.length === state.pairs.length) {
            state.pairs = state.pairs.map((pair, idx) => ({
                handle: {
                    x: data.updated_handles_image[idx].x,
                    y: data.updated_handles_image[idx].y,
                },
                target: pair.target,
            }));
            renderPairList();
            drawCanvas();
        }

        setStatus("Drag optimization complete. Output image and feature map are updated.", "info");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setBusy(false);
    }
});

refreshModeVisibility();
renderPairList();
drawCanvas();
checkHealth();
