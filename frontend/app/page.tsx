"use client";

import { useState } from "react";
import Link from "next/link";
import { Upload, FileSearch, ClipboardCheck, BarChart3, Brain, FlaskConical, Users, Globe } from "lucide-react";

export default function Home() {
  const [lang, setLang] = useState<"fr" | "en">("fr");

  const t = {
    fr: {
      title: "Système d'Aide au Diagnostic Radiologique",
      subtitle: "Segmentation sémantique et génération automatique de comptes rendus — Centre de Radiologie Emilie, Libreville",
      badge: "IA Médicale • Brouillon uniquement • Validation médicale requise",
      doctorTitle: "Espace Médecin",
      doctorDesc: "Uploadez des images DICOM, obtenez un brouillon de compte rendu IA, validez et téléchargez le PDF final.",
      doctorBtn: "Accéder →",
      patientTitle: "Espace Patient",
      patientDesc: "Uploadez vos images et consultez une analyse préliminaire IA. Résultat soumis à validation de votre radiologue.",
      patientBtn: "Accéder →",
      dashTitle: "Tableau de Bord",
      dashDesc: "Statistiques, performances du modèle IA, historique des examens.",
      dashBtn: "Voir →",
      how: "Comment ça fonctionne",
      step1: "Upload DICOM", step1d: "Glissez votre fichier .zip DICOM",
      step2: "Segmentation IA", step2d: "U-Net détecte les régions d'intérêt",
      step3: "Brouillon généré", step3d: "Compte rendu en français, style CRE",
      step4: "Validation médecin", step4d: "Le radiologue corrige et signe",
      disclaimer: "Ce système génère des brouillons IA à titre d'assistance uniquement. Tout compte rendu doit être validé par un radiologue qualifié avant usage clinique.",
    },
    en: {
      title: "AI-Assisted Radiology Reporting",
      subtitle: "Semantic segmentation and automated report generation — Centre de Radiologie Emilie, Libreville",
      badge: "Medical AI • Draft only • Medical validation required",
      doctorTitle: "Doctor Portal",
      doctorDesc: "Upload DICOM images, get an AI draft report, review, edit and download the final PDF.",
      doctorBtn: "Access →",
      patientTitle: "Patient Portal",
      patientDesc: "Upload your images and view a preliminary AI analysis. Results subject to radiologist validation.",
      patientBtn: "Access →",
      dashTitle: "Dashboard",
      dashDesc: "Statistics, AI model performance, exam history.",
      dashBtn: "View →",
      how: "How it works",
      step1: "Upload DICOM", step1d: "Drag your .zip DICOM file",
      step2: "AI Segmentation", step2d: "U-Net detects regions of interest",
      step3: "Draft generated", step3d: "Report in French, CRE style",
      step4: "Doctor validation", step4d: "Radiologist reviews and signs",
      disclaimer: "This system generates AI drafts for assistance purposes only. All reports must be validated by a qualified radiologist before clinical use.",
    },
  }[lang];

  return (
    <div className="min-h-screen" style={{ background: "var(--background)" }}>
      <div className="h-1" style={{ background: "var(--cre-blue)" }} />

      <header className="bg-white border-b shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
              style={{ background: "var(--cre-blue)" }}>CRE</div>
            <div>
              <div className="font-bold text-sm" style={{ color: "var(--cre-blue)" }}>CENTRE RADIOLOGIE EMILIE</div>
              <div className="text-xs text-gray-500">Libreville, Gabon — Radiology AI System</div>
            </div>
          </div>
          <button onClick={() => setLang(lang === "fr" ? "en" : "fr")}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 transition-colors">
            <Globe size={14} /> {lang === "fr" ? "EN" : "FR"}
          </button>
        </div>
      </header>

      <section className="py-14 px-6 text-center bg-white border-b">
        <div className="max-w-3xl mx-auto">
          <div className="inline-block mb-4 px-3 py-1 rounded-full text-xs font-semibold text-white"
            style={{ background: "var(--cre-red)" }}>{t.badge}</div>
          <h1 className="text-3xl font-bold mb-3" style={{ color: "var(--cre-blue)" }}>{t.title}</h1>
          <p className="text-gray-600 text-base leading-relaxed">{t.subtitle}</p>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid md:grid-cols-3 gap-6">
          <Link href="/doctor" className="group bg-white rounded-2xl border shadow-sm hover:shadow-md transition-all p-6">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4" style={{ background: "#e8f0fb" }}>
              <FlaskConical size={24} style={{ color: "var(--cre-blue)" }} />
            </div>
            <h2 className="font-bold text-lg mb-2" style={{ color: "var(--cre-blue)" }}>{t.doctorTitle}</h2>
            <p className="text-sm text-gray-600 mb-4 leading-relaxed">{t.doctorDesc}</p>
            <span className="text-sm font-semibold group-hover:underline" style={{ color: "var(--cre-blue)" }}>{t.doctorBtn}</span>
          </Link>

          <Link href="/patient" className="group bg-white rounded-2xl border shadow-sm hover:shadow-md transition-all p-6">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4" style={{ background: "#fdecea" }}>
              <Users size={24} style={{ color: "var(--cre-red)" }} />
            </div>
            <h2 className="font-bold text-lg mb-2" style={{ color: "var(--cre-blue)" }}>{t.patientTitle}</h2>
            <p className="text-sm text-gray-600 mb-4 leading-relaxed">{t.patientDesc}</p>
            <span className="text-sm font-semibold group-hover:underline" style={{ color: "var(--cre-red)" }}>{t.patientBtn}</span>
          </Link>

          <Link href="/dashboard" className="group bg-white rounded-2xl border shadow-sm hover:shadow-md transition-all p-6">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 bg-gray-100">
              <BarChart3 size={24} className="text-gray-600" />
            </div>
            <h2 className="font-bold text-lg mb-2" style={{ color: "var(--cre-blue)" }}>{t.dashTitle}</h2>
            <p className="text-sm text-gray-600 mb-4 leading-relaxed">{t.dashDesc}</p>
            <span className="text-sm font-semibold text-gray-600 group-hover:underline">{t.dashBtn}</span>
          </Link>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-12">
        <h2 className="text-lg font-bold mb-6 text-center" style={{ color: "var(--cre-blue)" }}>{t.how}</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: <Upload size={20} />, title: t.step1, desc: t.step1d },
            { icon: <Brain size={20} />, title: t.step2, desc: t.step2d },
            { icon: <FileSearch size={20} />, title: t.step3, desc: t.step3d },
            { icon: <ClipboardCheck size={20} />, title: t.step4, desc: t.step4d },
          ].map((s, i) => (
            <div key={i} className="bg-white rounded-xl border p-4 text-center">
              <div className="w-9 h-9 rounded-full text-white flex items-center justify-center mx-auto mb-3"
                style={{ background: "var(--cre-blue)" }}>{s.icon}</div>
              <div className="text-xs font-bold mb-1" style={{ color: "var(--cre-blue)" }}>{s.title}</div>
              <div className="text-xs text-gray-500">{s.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t bg-white">
        <div className="max-w-6xl mx-auto px-6 py-4 text-center text-xs text-gray-400">{t.disclaimer}</div>
      </footer>
    </div>
  );
}
