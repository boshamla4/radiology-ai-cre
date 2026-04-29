"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Activity, FileCheck, Clock, Percent, Monitor, Cpu } from "lucide-react";
import { getDashboard, DashboardStats } from "@/lib/api";

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboard().then(setStats).catch(console.error).finally(() => setLoading(false));
  }, []);

  const cards = stats ? [
    { label: "Examens total", value: stats.total_studies, icon: <Monitor size={20} />, color: "var(--cre-blue)" },
    { label: "Scanners (CT)", value: stats.ct_studies, icon: <Cpu size={20} />, color: "#0891b2" },
    { label: "IRM (MR)", value: stats.mr_studies, icon: <Activity size={20} />, color: "#7c3aed" },
    { label: "Rapports validés", value: stats.validated_reports, icon: <FileCheck size={20} />, color: "#16a34a" },
    { label: "Brouillons en attente", value: stats.ai_drafts_pending, icon: <Clock size={20} />, color: "#d97706" },
    {
      label: "Confiance IA moyenne",
      value: stats.avg_ai_confidence ? `${Math.round(stats.avg_ai_confidence * 100)}%` : "—",
      icon: <Percent size={20} />, color: "var(--cre-red)"
    },
  ] : [];

  return (
    <div className="min-h-screen" style={{ background: "var(--background)" }}>
      <div className="h-1" style={{ background: "var(--cre-blue)" }} />
      <header className="bg-white border-b shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-4">
          <Link href="/" className="text-gray-400 hover:text-gray-700"><ArrowLeft size={18} /></Link>
          <div className="font-bold" style={{ color: "var(--cre-blue)" }}>Tableau de Bord — Centre de Radiologie Emilie</div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {loading ? (
          <div className="text-center text-gray-400 py-20">Chargement des statistiques...</div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
              {cards.map((c, i) => (
                <div key={i} className="bg-white rounded-2xl border p-5">
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-9 h-9 rounded-xl flex items-center justify-center text-white"
                      style={{ background: c.color }}>
                      {c.icon}
                    </div>
                  </div>
                  <div className="text-2xl font-bold mb-1">{c.value}</div>
                  <div className="text-xs text-gray-500">{c.label}</div>
                </div>
              ))}
            </div>

            <div className="bg-white rounded-2xl border p-6">
              <h2 className="font-bold mb-4" style={{ color: "var(--cre-blue)" }}>
                Système IA — État
              </h2>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-500">Modèle de segmentation</span>
                  <span className="font-medium text-yellow-600">U-Net v1.0 (poids initiaux — entraînement requis)</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-500">Générateur de rapports</span>
                  <span className="font-medium">Ollama LLaVA (RAG + few-shot)</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-500">Base de données comptes rendus</span>
                  <span className="font-medium">{stats?.validated_reports || 0} cas indexés</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-500">Apprentissage continu</span>
                  <span className="font-medium text-green-600">Actif — {stats?.validated_reports || 0} corrections intégrées</span>
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
