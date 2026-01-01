/**
 * Symptoms Page - Stage B Symptom Form
 * Per docs/ui_flow_spec.md and docs/risk_copywriting_library.md
 */
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { assessTriage } from '../api/client'
import { AssessmentRequest, AssessmentResponse } from '../types'
import './Symptoms.css'

export default function Symptoms() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Load previous form data and response
  const [formData, setFormData] = useState<AssessmentRequest>(() => {
    const saved = localStorage.getItem('assessment_form_data')
    const savedResponse = localStorage.getItem('assessment_response')
    
    // Start with saved form data
    const base = saved ? JSON.parse(saved) : {}
    
    // If we have a previous response, use it as base
    if (savedResponse) {
      const response: AssessmentResponse = JSON.parse(savedResponse)
      // Merge with form data
      return { ...base }
    }
    
    return base
  })

  const [symptoms, setSymptoms] = useState({
    discomfort_eyestrain: formData.discomfort_eyestrain ?? 0,
    redness_in_eye: formData.redness_in_eye ?? 0,
    itchiness_irritation_in_eye: formData.itchiness_irritation_in_eye ?? 0,
  })

  const handleSymptomChange = (symptom: keyof typeof symptoms, value: number) => {
    setSymptoms((prev) => ({ ...prev, [symptom]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      // Merge symptoms with existing form data
      const triageRequest: AssessmentRequest = {
        ...formData,
        ...symptoms,
      }

      const result = await assessTriage(triageRequest)
      
      // Save updated form data with symptoms
      localStorage.setItem('assessment_form_data', JSON.stringify(triageRequest))
      
      // Save response to localStorage
      localStorage.setItem('assessment_response', JSON.stringify(result))
      
      // Navigate to result page
      navigate('/result')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Có lỗi xảy ra. Vui lòng thử lại.')
    } finally {
      setLoading(false)
    }
  }

  const handleSkip = () => {
    // Navigate back to result if we have a previous response
    const savedResponse = localStorage.getItem('assessment_response')
    if (savedResponse) {
      navigate('/result')
    } else {
      // Otherwise go back to assessment
      navigate('/quick-assessment')
    }
  }

  return (
    <div className="symptoms-page">
      <header className="page-header">
        <h1>Triệu chứng mắt</h1>
        <p className="subtitle">
          Để tăng độ chắc của phân loại, bạn có gặp các triệu chứng sau không?
        </p>
      </header>

      <form onSubmit={handleSubmit} className="symptoms-form">
        <div className="symptom-list">
          <div className="symptom-item">
            <label className="symptom-toggle">
              <input
                type="checkbox"
                checked={symptoms.discomfort_eyestrain === 1}
                onChange={(e) =>
                  handleSymptomChange('discomfort_eyestrain', e.target.checked ? 1 : 0)
                }
              />
              <div className="symptom-content">
                <strong>Khó chịu / Mỏi mắt</strong>
                <p className="symptom-description">
                  Cảm giác mỏi, khó chịu hoặc căng tức ở mắt sau khi nhìn màn hình hoặc đọc sách
                </p>
              </div>
            </label>
          </div>

          <div className="symptom-item">
            <label className="symptom-toggle">
              <input
                type="checkbox"
                checked={symptoms.redness_in_eye === 1}
                onChange={(e) =>
                  handleSymptomChange('redness_in_eye', e.target.checked ? 1 : 0)
                }
              />
              <div className="symptom-content">
                <strong>Đỏ mắt</strong>
                <p className="symptom-description">
                  Mắt có vẻ đỏ hoặc có vết đỏ, có thể kèm theo cảm giác nóng rát
                </p>
              </div>
            </label>
          </div>

          <div className="symptom-item">
            <label className="symptom-toggle">
              <input
                type="checkbox"
                checked={symptoms.itchiness_irritation_in_eye === 1}
                onChange={(e) =>
                  handleSymptomChange('itchiness_irritation_in_eye', e.target.checked ? 1 : 0)
                }
              />
              <div className="symptom-content">
                <strong>Ngứa / Kích ứng</strong>
                <p className="symptom-description">
                  Cảm giác ngứa, châm chích hoặc kích ứng ở mắt
                </p>
              </div>
            </label>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        <div className="form-actions">
          <button type="button" onClick={handleSkip} className="skip-button">
            Bỏ qua và xem kết quả hiện tại
          </button>
          <button type="submit" disabled={loading} className="submit-button primary">
            {loading ? 'Đang xử lý...' : 'Phân loại với triệu chứng'}
          </button>
        </div>
      </form>
    </div>
  )
}
