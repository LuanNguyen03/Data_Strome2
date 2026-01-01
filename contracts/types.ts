/**
 * TypeScript types matching contracts/schemas.py
 * Generated from Pydantic schemas for frontend type safety
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
  risk_A: number; // 0-100
  trigger_symptom: boolean;
}

export interface TriageInfo {
  prob_B: number; // 0-100
  triage_level: RiskLevel;
}

export interface AssessmentResponse {
  request_id: string;
  timestamp: string; // ISO8601
  mode_used: ModeUsed;
  risk_score: number; // 0-100
  risk_level: RiskLevel;
  confidence: Confidence;
  missing_fields: string[];
  top_factors: TopFactor[];
  next_step: NextStep;
  disclaimers: string[];
  screening?: ScreeningInfo;
  triage?: TriageInfo;
}

export interface AssessmentRequest {
  // Person
  age?: number; // 18-45
  gender?: number; // 0=F, 1=M
  height?: number; // 120-230
  weight?: number; // 30-250

  // Sleep
  sleep_duration?: number; // 0-24
  sleep_quality?: number; // 1-5
  sleep_disorder?: number; // 0-1
  wake_up_during_night?: number; // 0-1
  feel_sleepy_during_day?: number; // 0-1

  // Device/Screen
  average_screen_time?: number; // 0-24
  smart_device_before_bed?: number; // 0-1
  bluelight_filter?: number; // 0-1

  // Lifestyle
  stress_level?: number; // 1-5
  daily_steps?: number; // 0-50000
  physical_activity?: number; // 0-600
  caffeine_consumption?: number; // 0-1
  alcohol_consumption?: number; // 0-1
  smoking?: number; // 0-1

  // Vitals
  systolic?: number; // 70-250
  diastolic?: number; // 40-150
  heart_rate?: number; // 40-220

  // Medical
  medical_issue?: number; // 0-1
  ongoing_medication?: number; // 0-1

  // Symptoms (Stage B only)
  discomfort_eyestrain?: number; // 0-1
  redness_in_eye?: number; // 0-1
  itchiness_irritation_in_eye?: number; // 0-1
}

