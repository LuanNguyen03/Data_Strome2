import axios from 'axios'
import { AssessmentRequest, AssessmentResponse } from '../types'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const assessScreening = async (
  request: AssessmentRequest
): Promise<AssessmentResponse> => {
  const response = await client.post<AssessmentResponse>('/api/v1/assessments/screening', request)
  return response.data
}

export const assessTriage = async (
  request: AssessmentRequest
): Promise<AssessmentResponse> => {
  const response = await client.post<AssessmentResponse>('/api/v1/assessments/triage', request)
  return response.data
}

export const healthCheck = async (): Promise<{ status: string; service: string; version: string }> => {
  const response = await client.get('/api/v1/healthz')
  return response.data
}

export const listKPIs = async (): Promise<any> => {
  const response = await client.get('/api/v1/olap/kpis')
  return response.data
}

export const getKPI = async (name: string, page: number = 1, pageSize: number = 100): Promise<any> => {
  const response = await client.get(`/api/v1/olap/kpis/${name}`, {
    params: { page, page_size: pageSize }
  })
  return response.data
}

