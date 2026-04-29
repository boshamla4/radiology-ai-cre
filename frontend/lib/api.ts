import axios from "axios";

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 120000,
});

export interface DicomMetadata {
  patient_id: string;
  patient_name: string;
  birth_date: string;
  age: string;
  sex: string;
  modality: "CT" | "MR" | "OTHER";
  study_date: string;
  study_description: string;
  series_description: string;
  protocol_name: string;
  body_part: string | null;
  institution: string;
  accession_number: string;
  slice_count: number;
}

export interface UploadResponse {
  study_id: number;
  metadata: DicomMetadata;
  message: string;
}

export interface FindingItem {
  structure: string;
  location: string | null;
  size_mm: number | null;
  description: string;
  is_pathological: boolean;
  confidence: number;
}

export interface SegmentationResult {
  findings: FindingItem[];
  overlay_image_base64: string;
  model_version: string;
  dice_score: number | null;
}

export interface ReportDraft {
  report_id: number;
  status: string;
  exam_type: string;
  technique: string;
  ai_findings: string;
  ai_conclusion: string;
  segmentation: SegmentationResult | null;
  similar_cases_count: number;
  ai_confidence: number;
  created_at: string;
}

export interface DashboardStats {
  total_studies: number;
  ct_studies: number;
  mr_studies: number;
  validated_reports: number;
  ai_drafts_pending: number;
  avg_ai_confidence: number | null;
  pathologies_found: number;
  normal_exams: number;
}

export const uploadDicom = async (file: File): Promise<UploadResponse> => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<UploadResponse>("/upload/dicom", form);
  return data;
};

export const generateDraft = async (
  study_id: number,
  indication: string,
  prescribing_doctor: string
): Promise<ReportDraft> => {
  const { data } = await api.post<ReportDraft>("/analysis/generate-draft", {
    study_id,
    indication,
    prescribing_doctor,
  });
  return data;
};

export const validateReport = async (
  report_id: number,
  radiologist: string,
  final_findings: string,
  final_conclusion: string
) => {
  const { data } = await api.post("/reports/validate", {
    report_id,
    radiologist,
    final_findings,
    final_conclusion,
  });
  return data;
};

export const getDashboard = async (): Promise<DashboardStats> => {
  const { data } = await api.get<DashboardStats>("/reports/dashboard");
  return data;
};
