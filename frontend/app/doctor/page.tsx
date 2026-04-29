"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import {
  Upload, ArrowLeft, CheckCircle, AlertTriangle,
  FileText, Eye, Download, Loader2, ChevronDown, ChevronUp
} from "lucide-react";
import { uploadDicom, generateDraft, validateReport, DicomMetadata, ReportDraft } from "@/lib/api";

type Step = "upload" | "info" | "analysis" | "review" | "done";

export default function DoctorPage() {
  const [step, setStep] = useState<Step>("upload");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Data
  const [studyId, setStudyId] = useState<number | null>(null);
  const [metadata, setMetadata] = useState<DicomMetadata | null>(null);
  const [indication, setIndication] = useState("");
  const [prescribingDoctor, setPrescribingDoctor] = useState("");
  const [draft, setDraft] = useState<ReportDraft | null>(null);
  const [editedFindings, setEditedFindings] = useState("");
  const [editedConclusion, setEditedConclusion] = useState("");
  const [radiologist, setRadiologist] = useState("");
  const [reportId, setReportId] = useState<number | null>(null);
  const [showOverlay, setShowOverlay] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".zip")) {
      setError("Veuillez sélectionner un fichier .zip contenant les images DICOM.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await uploadDicom(file);
      setStudyId(res.study_id);
      setMetadata(res.metadata);
      setStep("info");
    } catch (e: unknown) {
      setError("Erreur lors de l'upload. Vérifiez que le fichier est un archive DICOM valide.");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleAnalyse = async () => {
    if (!studyId || !indication || !prescribingDoctor) return;
    setLoading(true);
    setError("");
    try {
      const d = await generateDraft(studyId, indication, prescribingDoctor);
      setDraft(d);
      setEditedFindings(d.ai_findings);
      setEditedConclusion(d.ai_conclusion);
      setStep("review");
    } catch {
      setError("Erreur lors de la génération du brouillon IA.");
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    if (!draft || !radiologist) return;
    setLoading(true);
    try {
      const res = await validateReport(draft.report_id, radiologist, editedFindings, editedConclusion);
      setReportId(res.report_id);
      setStep("done");
    } catch {
      setError("Erreur lors de la validation.");
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = (c: number) =>
    c >= 0.75 ? "text-green-600" : c >= 0.55 ? "text-yellow-600" : "text-red-500";

  return (
    <div className="min-h-screen" style={{ background: "var(--background)" }}>
      <div className="h-1" style={{ background: "var(--cre-blue)" }} />

      <header className="bg-white border-b shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-4">
          <Link href="/" className="text-gray-400 hover:text-gray-700 transition-colors">
            <ArrowLeft size={18} />
          </Link>
          <div className="font-bold" style={{ color: "var(--cre-blue)" }}>
            Espace Médecin — Analyse DICOM
          </div>
          {/* Step indicator */}
          <div className="ml-auto flex items-center gap-2 text-xs text-gray-400">
            {(["upload","info","review","done"] as Step[]).map((s, i) => (
              <span key={s} className={`flex items-center gap-1 ${step === s ? "text-blue-700 font-semibold" : ""}`}>
                {i > 0 && <span className="mx-1">›</span>}
                {["Upload","Informations","Révision","Terminé"][i]}
              </span>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 p-3 rounded-lg bg-red-50 border border-red-200 flex items-center gap-2 text-sm text-red-700">
            <AlertTriangle size={16} /> {error}
          </div>
        )}

        {/* STEP 1: Upload */}
        {step === "upload" && (
          <div
            className={`border-2 border-dashed rounded-2xl p-16 text-center transition-all cursor-pointer ${
              dragging ? "border-blue-400 bg-blue-50" : "border-gray-200 bg-white hover:border-blue-300"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById("fileInput")?.click()}
          >
            {loading ? (
              <div className="flex flex-col items-center gap-3">
                <Loader2 size={40} className="animate-spin" style={{ color: "var(--cre-blue)" }} />
                <p className="text-gray-500">Lecture des fichiers DICOM...</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full flex items-center justify-center"
                  style={{ background: "#e8f0fb" }}>
                  <Upload size={28} style={{ color: "var(--cre-blue)" }} />
                </div>
                <div>
                  <p className="font-semibold text-lg mb-1" style={{ color: "var(--cre-blue)" }}>
                    Glissez votre archive DICOM ici
                  </p>
                  <p className="text-sm text-gray-500">ou cliquez pour sélectionner un fichier <strong>.zip</strong></p>
                  <p className="text-xs text-gray-400 mt-2">Fichiers DICOM CT ou IRM, max 500 MB</p>
                </div>
              </div>
            )}
            <input id="fileInput" type="file" accept=".zip" className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
          </div>
        )}

        {/* STEP 2: Patient info form */}
        {step === "info" && metadata && (
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl border p-6">
              <h2 className="font-bold text-base mb-4" style={{ color: "var(--cre-blue)" }}>
                Informations Patient (extraites du DICOM)
              </h2>
              <div className="space-y-3 text-sm">
                {[
                  ["Patient", metadata.patient_name],
                  ["ID", metadata.patient_id],
                  ["Date de naissance", metadata.birth_date],
                  ["Âge", metadata.age],
                  ["Sexe", metadata.sex === "M" ? "Masculin" : metadata.sex === "F" ? "Féminin" : metadata.sex],
                  ["Modalité", metadata.modality],
                  ["Région", metadata.body_part || "—"],
                  ["Protocole", metadata.protocol_name],
                  ["Date examen", metadata.study_date],
                  ["Coupes", String(metadata.slice_count)],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-gray-500">{k}</span>
                    <span className="font-medium text-right max-w-[60%] truncate">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-2xl border p-6 flex flex-col gap-4">
              <h2 className="font-bold text-base mb-2" style={{ color: "var(--cre-blue)" }}>
                Informations Cliniques
              </h2>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Médecin prescripteur <span className="text-red-500">*</span>
                </label>
                <input
                  className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
                  placeholder="Dr. NOM Prénom"
                  value={prescribingDoctor}
                  onChange={(e) => setPrescribingDoctor(e.target.value)}
                />
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Indication clinique <span className="text-red-500">*</span>
                </label>
                <textarea
                  className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300 h-28 resize-none"
                  placeholder="Ex: Céphalée chronique, ATCD d'AVC ischémique..."
                  value={indication}
                  onChange={(e) => setIndication(e.target.value)}
                />
              </div>
              <button
                onClick={handleAnalyse}
                disabled={!indication || !prescribingDoctor || loading}
                className="w-full py-3 rounded-xl text-white font-semibold text-sm transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                style={{ background: "var(--cre-blue)" }}
              >
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyse en cours...</> : "Lancer l'analyse IA →"}
              </button>
              {loading && (
                <p className="text-xs text-center text-gray-400">
                  Segmentation U-Net + génération du compte rendu... (~30-60s)
                </p>
              )}
            </div>
          </div>
        )}

        {/* STEP 3: Review & edit */}
        {step === "review" && draft && metadata && (
          <div className="space-y-6">
            {/* Confidence + segmentation */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-2xl border p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-bold text-sm" style={{ color: "var(--cre-blue)" }}>
                    Résultats de Segmentation IA
                  </h3>
                  <span className={`text-sm font-bold ${confidenceColor(draft.ai_confidence)}`}>
                    Confiance: {Math.round(draft.ai_confidence * 100)}%
                  </span>
                </div>
                {draft.segmentation?.findings?.length ? (
                  <ul className="space-y-2">
                    {draft.segmentation.findings.map((f, i) => (
                      <li key={i} className={`text-xs p-2 rounded-lg ${f.is_pathological ? "bg-red-50 border border-red-200" : "bg-gray-50"}`}>
                        <span className={`font-semibold ${f.is_pathological ? "text-red-700" : "text-gray-700"}`}>
                          {f.structure}
                        </span>
                        {f.size_mm && <span className="text-gray-500"> — {f.size_mm}mm</span>}
                        <span className="block text-gray-500 mt-0.5">{f.description}</span>
                      </li>
                    ))}
                  </ul>
                ) : <p className="text-xs text-gray-400">Aucune structure détectée (modèle non entraîné).</p>}
                <p className="text-xs text-gray-400 mt-3">
                  {draft.similar_cases_count} cas similaires récupérés • {draft.segmentation?.model_version}
                </p>
              </div>

              {draft.segmentation?.overlay_image_base64 && (
                <div className="bg-white rounded-2xl border p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-bold text-sm" style={{ color: "var(--cre-blue)" }}>
                      Overlay de Segmentation
                    </h3>
                    <button onClick={() => setShowOverlay(!showOverlay)}
                      className="text-xs text-blue-600 flex items-center gap-1">
                      {showOverlay ? <><ChevronUp size={12} /> Masquer</> : <><ChevronDown size={12} /> Afficher</>}
                    </button>
                  </div>
                  {showOverlay && (
                    <img
                      src={`data:image/png;base64,${draft.segmentation.overlay_image_base64}`}
                      alt="Segmentation overlay"
                      className="rounded-lg w-full object-contain max-h-64"
                    />
                  )}
                </div>
              )}
            </div>

            {/* Editable report */}
            <div className="bg-white rounded-2xl border p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="px-2 py-0.5 rounded text-xs font-bold text-white" style={{ background: "var(--cre-red)" }}>
                  BROUILLON IA
                </div>
                <h3 className="font-bold text-sm" style={{ color: "var(--cre-blue)" }}>
                  {draft.exam_type} — À réviser avant validation
                </h3>
              </div>

              <div className="text-xs text-gray-500 mb-1 font-medium">Technique</div>
              <p className="text-sm text-gray-700 mb-4 p-2 bg-gray-50 rounded">{draft.technique}</p>

              <div className="text-xs text-gray-500 mb-1 font-medium">RÉSULTAT (modifiable)</div>
              <textarea
                className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300 min-h-36 resize-y mb-4"
                value={editedFindings}
                onChange={(e) => setEditedFindings(e.target.value)}
              />

              <div className="text-xs text-gray-500 mb-1 font-medium">CONCLUSION (modifiable)</div>
              <textarea
                className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300 min-h-20 resize-y mb-4"
                value={editedConclusion}
                onChange={(e) => setEditedConclusion(e.target.value)}
              />

              <div className="mb-4">
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Votre nom (Radiologue) <span className="text-red-500">*</span>
                </label>
                <input
                  className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
                  placeholder="Dr. NOM Prénom"
                  value={radiologist}
                  onChange={(e) => setRadiologist(e.target.value)}
                />
              </div>

              <div className="flex gap-3">
                <a
                  href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/reports/draft-pdf/${draft.report_id}`}
                  target="_blank"
                  className="flex items-center gap-2 px-4 py-2 rounded-xl border text-sm font-medium hover:bg-gray-50 transition"
                >
                  <Eye size={15} /> Aperçu PDF brouillon
                </a>
                <button
                  onClick={handleValidate}
                  disabled={!radiologist || loading}
                  className="flex-1 py-2 rounded-xl text-white font-semibold text-sm flex items-center justify-center gap-2 disabled:opacity-50 transition"
                  style={{ background: "var(--cre-blue)" }}
                >
                  {loading ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle size={15} />}
                  Valider et générer le PDF final
                </button>
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: Done */}
        {step === "done" && reportId && (
          <div className="bg-white rounded-2xl border p-12 text-center">
            <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-4">
              <CheckCircle size={32} className="text-green-600" />
            </div>
            <h2 className="text-xl font-bold mb-2" style={{ color: "var(--cre-blue)" }}>
              Compte rendu validé !
            </h2>
            <p className="text-gray-500 text-sm mb-8">
              Le PDF final a été généré et archivé. Le brouillon IA a été marqué comme données d'entraînement.
            </p>
            <div className="flex gap-3 justify-center">
              <a
                href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/reports/download/${reportId}`}
                target="_blank"
                className="flex items-center gap-2 px-6 py-3 rounded-xl text-white font-semibold text-sm"
                style={{ background: "var(--cre-blue)" }}
              >
                <Download size={16} /> Télécharger le PDF
              </a>
              <button onClick={() => {
                setStep("upload"); setMetadata(null); setDraft(null);
                setIndication(""); setPrescribingDoctor(""); setRadiologist("");
              }}
                className="px-6 py-3 rounded-xl border font-semibold text-sm hover:bg-gray-50 transition">
                Nouvel examen
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
