// =====================================================
// DOM ELEMENTS
// =====================================================
const analyzeBtn = document.getElementById("analyzeBtn");
const downloadPdfBtn = document.getElementById("downloadPdfBtn");
const fileInput = document.getElementById("imageInput");

const loading = document.getElementById("loading");
const resultDiv = document.getElementById("result");

// Text outputs
const boneEl = document.getElementById("bone");
const fractureEl = document.getElementById("fracture");
const confidenceEl = document.getElementById("confidence");
const angleEl = document.getElementById("angle");
const severityEl = document.getElementById("severity");
const recommendationEl = document.getElementById("recommendation");

// Images
const roiImg = document.getElementById("roiImage");
const gradcamImg = document.getElementById("gradcamImage");

// =====================================================
// STATE
// =====================================================
let roiBase64 = null;
let gradcamBase64 = null;

// =====================================================
// INITIAL STATE
// =====================================================
analyzeBtn.disabled = true;
downloadPdfBtn.disabled = true;
loading.classList.add("hidden");
resultDiv.classList.add("hidden");

// =====================================================
// FILE INPUT
// =====================================================
fileInput.addEventListener("change", () => {
  analyzeBtn.disabled = fileInput.files.length === 0;
  resultDiv.classList.add("hidden");
  downloadPdfBtn.disabled = true;

  roiBase64 = null;
  gradcamBase64 = null;
});

// =====================================================
// ANALYZE BUTTON
// =====================================================
analyzeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  analyzeBtn.disabled = true;
  loading.classList.remove("hidden");
  resultDiv.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Prediction request failed");
    }

    const data = await response.json();

    // =============================
    // TEXT OUTPUTS
    // =============================
    boneEl.textContent = data.bone_type || "N/A";
    fractureEl.textContent = data.fracture || "N/A";

    confidenceEl.textContent =
      typeof data.confidence === "number"
        ? `${(data.confidence * 100).toFixed(1)}%`
        : "N/A";

    angleEl.textContent =
      data.displacement_angle !== null &&
      data.displacement_angle !== undefined
        ? data.displacement_angle
        : "N/A";

    severityEl.textContent = data.severity || "N/A";
    recommendationEl.textContent = data.recommendation || "N/A";

    // =============================
    // ROI IMAGE
    // =============================
    if (data.roi_image) {
      roiBase64 = data.roi_image;
      roiImg.src = `data:image/png;base64,${roiBase64}`;
      roiImg.style.display = "block";
    } else {
      roiImg.style.display = "none";
      roiBase64 = null;
    }

    // =============================
    // GRAD-CAM IMAGE
    // =============================
    if (data.gradcam_heatmap) {
      gradcamBase64 = data.gradcam_heatmap;
      gradcamImg.src = `data:image/png;base64,${gradcamBase64}`;
      gradcamImg.style.display = "block";
    } else {
      gradcamImg.style.display = "none";
      gradcamBase64 = null;
    }

    loading.classList.add("hidden");
    resultDiv.classList.remove("hidden");
    downloadPdfBtn.disabled = false;

  } catch (error) {
    console.error(error);
    loading.classList.add("hidden");
    alert("Prediction failed. Please check backend logs.");
  } finally {
    analyzeBtn.disabled = false;
  }
});

// =====================================================
// DOWNLOAD PDF (FRONTEND GENERATED – STABLE)
// =====================================================
downloadPdfBtn.addEventListener("click", () => {
  if (!window.jspdf || !window.jspdf.jsPDF) {
    alert("PDF library not loaded.");
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  let y = 18;

  // =============================
  // HEADER
  // =============================
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text("AI-Based Bone Fracture Clinical Report", 105, y, { align: "center" });

  y += 10;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.text(`Report Date: ${new Date().toLocaleString()}`, 14, y);
  doc.text(`Report ID: AI-BFD-${Date.now()}`, 140, y);

  y += 10;
  doc.line(14, y, 196, y);
  y += 8;

  // =============================
  // CLINICAL FINDINGS
  // =============================
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text("Clinical Findings", 14, y);
  y += 6;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.text(`Anatomical Region: ${boneEl.textContent}`, 14, y); y += 6;
  doc.text(`Fracture Status: ${fractureEl.textContent}`, 14, y); y += 6;
  doc.text(`Confidence Score: ${confidenceEl.textContent}`, 14, y); y += 6;
  doc.text(`Displacement Angle: ${angleEl.textContent}°`, 14, y); y += 6;
  doc.text(`Severity Level: ${severityEl.textContent}`, 14, y); y += 10;

  // =============================
  // RECOMMENDATION
  // =============================
  doc.setFont("helvetica", "bold");
  doc.text("Clinical Recommendation", 14, y);
  y += 6;

  doc.setFont("helvetica", "normal");
  const recText = doc.splitTextToSize(recommendationEl.textContent, 170);
  doc.text(recText, 14, y);
  y += recText.length * 6 + 10;

  // =============================
  // VISUAL EVIDENCE
  // =============================
  if (roiBase64 || gradcamBase64) {
    doc.addPage();
    y = 18;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Visual Evidence", 14, y);
    y += 8;

    if (roiBase64) {
      doc.setFontSize(12);
      doc.text("Region of Interest (ROI)", 14, y);
      y += 4;
      doc.addImage(`data:image/png;base64,${roiBase64}`, "PNG", 14, y, 180, 70);
      y += 78;
    }

    if (gradcamBase64) {
      doc.setFontSize(12);
      doc.text("Grad-CAM Heatmap", 14, y);
      y += 4;
      doc.addImage(`data:image/png;base64,${gradcamBase64}`, "PNG", 14, y, 180, 70);
    }
  }

  // =============================
  // DISCLAIMER
  // =============================
  doc.addPage();
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text("Disclaimer", 14, 30);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.text(
    "This report is generated using an AI-assisted diagnostic system and is intended "
    + "to support clinical decision-making only. It must not replace professional "
    + "medical judgment. Final diagnosis and treatment decisions should be made by a "
    + "qualified healthcare professional.",
    14,
    40,
    { maxWidth: 180 }
  );

  // =============================
  // SAVE
  // =============================
  doc.save("AI_Bone_Fracture_Clinical_Report.pdf");
});