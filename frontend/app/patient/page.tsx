"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { Upload, ArrowLeft, AlertTriangle, Loader2, Info, Brain } from "lucide-react";
import { uploadDicom, generateDraft, DicomMetadata, ReportDraft } from "@/lib/api";

export default function PatientPage() {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [metadata, setMetadata] = useState<DicomMetadata | null>(null);
  const [indication, setIndication] = useState("");
  const [draft, setDraft] = useState<ReportDraft | null>(null);
  const [studyId, setStudyId] = useState<number | null>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".zip")) {
      setError("Veuillez sélectionner un fichier .zip contenant les images DICOM.");
      return;
    }
    setError(""); setLoading(true);
    try {
      const res = await uploadDicom(file);
      setStudyId(res.study_id);
      setMetadata(res.metadata);
    } catch {
      setError("Erreur lors de l'upload. Vérifiez votre fichier.");
    } finally { setLoading(false); }
  }, []);

  const handleAnalyse = async () => {
    if (!studyId) return;
    setLoading(true); setError("");
    try {
      const d = await generateDraft(studyId, indication || "Analyse demandée par le patient", "À compléter par le médecin");
      setDraft(d);
    } catch { setError("Erreur lors de l'analyse."); }
    finally { setLoading(false); }
  };

  return (
    <div className="min-h-screen" style={{ background: "var(--background)" }}>
      <div className="h-1" style={{ background: "var(--cre-red)" }} />
      <header className="bg-white border-b shadow-sm">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center gap-4">
          <Link href="/" className="text-gray-400 hover:text-gray-700"><ArrowLeft size={18} /></Link>
          <div className="font-bold" style={{ color: "var(--cre-blue)" }}>Espace Patient</div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-8 space-y-6">
        {/* Disclaimer banner */}
        <div className="p-4 rounded-xl border flex gap-3 text-sm" style={{ background: "#fff8e1", borderColor: "#f59e0b" }}>
          <Info size={16} className="mt-0.5 flex-shrink-0" style={{ color: "#f59e0b" }} />
          <div>
            <strong>Information importante :</strong> Cette analyse est générée par intelligence artificielle et constitue uniquement un brouillon préliminaire.
            Elle <strong>ne remplace pas</strong> l'avis d'un radiologue qualifié. Consultez votre médecin pour la validation de ces résultats.
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-red-50 border border-red-200 flex items-center gap-2 text-sm text-red-700">
            <AlertTriangle size={16} /> {error}
          </div>
        )}

        {/* Upload */}
        {!metadata && (
          <div
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer bg-white transition-all ${
              dragging ? "border-red-400 bg-red-50" : "border-gray-200 hover:border-red-300"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
            onClick={() => document.getElementById("patientFile")?.click()}
          >
            {loading ? (
              <div className="flex flex-col items-center gap-3">
                <Loader2 size={36} className="animate-spin" style={{ color: "var(--cre-red)" }} />
                <p className="text-gray-500 text-sm">Lecture de vos images médicales...</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: "#fdecea" }}>
                  <Upload size={24} style={{ color: "var(--cre-red)" }} />
                </div>
                <div>
                  <p className="font-semibold" style={{ color: "var(--cre-blue)" }}>Uploadez votre archive DICOM</p>
                  <p className="text-sm text-gray-500 mt-1">Fichier .zip fourni par votre centre d'imagerie</p>
                </div>
              </div>
            )}
            <input id="patientFile" type="file" accept=".zip" className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
          </div>
        )}

        {/* Patient info + analyse */}
        {metadata && !draft && (
          <div className="bg-white rounded-2xl border p-6 space-y-4">
            <h2 className="font-bold" style={{ color: "var(--cre-blue)" }}>
              Images reçues — {metadata.slice_count} coupes détectées
            </h2>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {[
                ["Patient", metadata.patient_name],
                ["Examen", metadata.study_description || metadata.modality],
                ["Date", metadata.study_date],
                ["Région", metadata.body_part || "—"],
              ].map(([k, v]) => (
                <div key={k} className="bg-gray-50 rounded-lg p-2">
                  <div className="text-xs text-gray-400">{k}</div>
                  <div className="font-medium text-xs">{v}</div>
                </div>
              ))}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Motif de l'examen (optionnel)
              </label>
              <input className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2"
                placeholder="Ex: douleur, suivi..." value={indication}
                onChange={(e) => setIndication(e.target.value)} />
            </div>
            <button onClick={handleAnalyse} disabled={loading}
              className="w-full py-3 rounded-xl text-white font-semibold text-sm flex items-center justify-center gap-2 disabled:opacity-50"
              style={{ background: "var(--cre-red)" }}>
              {loading ? <><Loader2 size={15} className="animate-spin" /> Analyse en cours...</> :
                <><Brain size={15} /> Analyser mes images</>}
            </button>
          </div>
        )}

        {/* AI results */}
        {draft && (
          <div className="space-y-4">
            <div className="p-3 rounded-xl border text-xs font-semibold text-center text-white"
              style={{ background: "var(--cre-red)" }}>
              ⚠ ANALYSE PRÉLIMINAIRE IA — NON VALIDÉE PAR UN RADIOLOGUE
            </div>
            <div className="bg-white rounded-2xl border p-6">
              <h3 className="font-bold mb-2" style={{ color: "var(--cre-blue)" }}>{draft.exam_type}</h3>
              <div className="text-xs text-gray-500 mb-1 font-medium">Observations préliminaires</div>
              <div className="text-sm text-gray-700 whitespace-pre-line bg-gray-50 rounded-lg p-3 mb-4">
                {draft.ai_findings}
              </div>
              <div className="text-xs text-gray-500 mb-1 font-medium">Conclusion préliminaire</div>
              <div className="text-sm font-semibold text-gray-800 bg-gray-50 rounded-lg p-3">
                {draft.ai_conclusion}
              </div>
            </div>
            <div className="p-4 rounded-xl bg-blue-50 border border-blue-200 text-sm text-blue-800">
              <strong>Prochaine étape :</strong> Partagez ce résultat avec votre médecin ou radiologue pour validation.
              Le compte rendu officiel sera établi après consultation médicale.
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
