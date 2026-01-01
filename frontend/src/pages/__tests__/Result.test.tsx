/**
 * Frontend Component Test - Result Page Disclaimers
 * Per docs/clinical_governance_checklist.md
 * Tests that disclaimers always render on Result page
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import Result from '../Result'
import { AssessmentResponse, ModeUsed, RiskLevel, Confidence } from '../../types'

// Mock localStorage
const mockLocalStorage = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
})

// Mock useNavigate
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Result Page - Disclaimers Always Render', () => {
  beforeEach(() => {
    mockLocalStorage.clear()
    mockNavigate.mockClear()
    vi.clearAllMocks()
  })

  const createMockResponse = (mode: ModeUsed): AssessmentResponse => ({
    request_id: 'test-123',
    timestamp: '2025-12-31T12:00:00Z',
    mode_used: mode,
    risk_score: 65.0,
    risk_level: RiskLevel.MEDIUM,
    confidence: Confidence.MEDIUM,
    missing_fields: [],
    top_factors: [],
    next_step: {
      title: 'Test next step',
      actions: ['Action 1'],
      ask_for_more_info: [],
      urgency: 'monitor' as const,
    },
    disclaimers: [
      'Kết quả chỉ hỗ trợ sàng lọc/triage, không thay thế chẩn đoán.',
      'Nếu triệu chứng kéo dài hoặc nặng, nên tham khảo bác sĩ.',
    ],
    ...(mode === ModeUsed.A_ONLY_SCREENING
      ? {
          screening: {
            risk_A: 65.0,
            trigger_symptom: true,
          },
        }
      : {
          triage: {
            prob_B: 0.65,
            triage_level: RiskLevel.MEDIUM,
          },
        }),
  })

  it('should always render disclaimers for Stage A response', () => {
    const response = createMockResponse(ModeUsed.A_ONLY_SCREENING)
    mockLocalStorage.setItem('assessment_response', JSON.stringify(response))

    render(
      <BrowserRouter>
        <Result />
      </BrowserRouter>
    )

    // Check disclaimers section exists
    const disclaimersSection = screen.getByText('Lưu ý quan trọng')
    expect(disclaimersSection).toBeInTheDocument()

    // Check disclaimer content
    const disclaimer1 = screen.getByText(/Kết quả chỉ hỗ trợ sàng lọc\/triage/)
    expect(disclaimer1).toBeInTheDocument()

    const disclaimer2 = screen.getByText(/Nếu triệu chứng kéo dài hoặc nặng/)
    expect(disclaimer2).toBeInTheDocument()
  })

  it('should always render disclaimers for Stage B response', () => {
    const response = createMockResponse(ModeUsed.B_WITH_SYMPTOMS)
    mockLocalStorage.setItem('assessment_response', JSON.stringify(response))

    render(
      <BrowserRouter>
        <Result />
      </BrowserRouter>
    )

    // Check disclaimers section exists
    const disclaimersSection = screen.getByText('Lưu ý quan trọng')
    expect(disclaimersSection).toBeInTheDocument()

    // Check disclaimer content
    const disclaimer1 = screen.getByText(/Kết quả chỉ hỗ trợ sàng lọc\/triage/)
    expect(disclaimer1).toBeInTheDocument()
  })

  it('should render all disclaimers from response', () => {
    const response: AssessmentResponse = {
      ...createMockResponse(ModeUsed.A_ONLY_SCREENING),
      disclaimers: [
        'Disclaimer 1',
        'Disclaimer 2',
        'Disclaimer 3',
      ],
    }
    mockLocalStorage.setItem('assessment_response', JSON.stringify(response))

    render(
      <BrowserRouter>
        <Result />
      </BrowserRouter>
    )

    // Check all disclaimers are rendered
    expect(screen.getByText('Disclaimer 1')).toBeInTheDocument()
    expect(screen.getByText('Disclaimer 2')).toBeInTheDocument()
    expect(screen.getByText('Disclaimer 3')).toBeInTheDocument()
  })

  it('should render disclaimers section even with empty disclaimers array', () => {
    const response: AssessmentResponse = {
      ...createMockResponse(ModeUsed.A_ONLY_SCREENING),
      disclaimers: [],
    }
    mockLocalStorage.setItem('assessment_response', JSON.stringify(response))

    render(
      <BrowserRouter>
        <Result />
      </BrowserRouter>
    )

    // Disclaimers section should still exist (though empty)
    const disclaimersSection = screen.getByText('Lưu ý quan trọng')
    expect(disclaimersSection).toBeInTheDocument()
  })
})
