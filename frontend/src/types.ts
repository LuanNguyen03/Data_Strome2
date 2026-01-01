/**
 * TypeScript types - matches contracts/types.ts
 */
export enum ModeUsed {
  A_ONLY_SCREENING = "A_only_screening",
  B_WITH_SYMPTOMS = "B_with_symptoms",
}

export enum RiskLevel {
  LOW = "Low",
  MEDIUM = "Medium",
  HIGH = "High",
}

export enum Confidence {
  HIGH = "High",
  MEDIUM = "Medium",
  LOW = "Low",
}

export enum Direction {
  INCREASE_RISK = "increase_risk",
  DECREASE_RISK = "decrease_risk",
  UNKNOWN = "unknown",
}

export enum Strength {
  HIGH = "High",
  MEDIUM = "Medium",
  LOW = "Low",
}

export enum Urgency {
  NONE = "none",
  MONITOR = "monitor",
  CONSIDER_VISIT = "consider_visit",
  VISIT_RECOMMENDED = "visit_recommended",
}

export interface TopFactor {
  feature: string;
  direction: Direction;
  strength: Strength;
  note: string;
}

export interface NextStep {
  title: string;
  actions: string[];
  ask_for_more_info: string[];
  urgency: Urgency;
}

export interface ScreeningInfo {
  risk_A: number;
  trigger_symptom: boolean;
}

export interface TriageInfo {
  prob_B: number;
  triage_level: RiskLevel;
}

export interface AssessmentResponse {
  request_id: string;
  timestamp: string;
  mode_used: ModeUsed;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: Confidence;
  missing_fields: string[];
  top_factors: TopFactor[];
  next_step: NextStep;
  disclaimers: string[];
  screening?: ScreeningInfo;
  triage?: TriageInfo;
  treatment_recommendations?: string;
  model_version?: string; // Added by backend per contract
}

export interface AssessmentRequest {
  age?: number;
  gender?: number;
  height?: number;
  weight?: number;
  sleep_duration?: number;
  sleep_quality?: number;
  sleep_disorder?: number;
  wake_up_during_night?: number;
  feel_sleepy_during_day?: number;
  average_screen_time?: number;
  smart_device_before_bed?: number;
  bluelight_filter?: number;
  stress_level?: number;
  daily_steps?: number;
  physical_activity?: number;
  caffeine_consumption?: number;
  alcohol_consumption?: number;
  smoking?: number;
  systolic?: number;
  diastolic?: number;
  heart_rate?: number;
  medical_issue?: number;
  ongoing_medication?: number;
  discomfort_eyestrain?: number;
  redness_in_eye?: number;
  itchiness_irritation_in_eye?: number;
}

