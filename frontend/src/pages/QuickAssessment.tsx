/**
 * Quick Assessment Page - Stage A Screening Form
 * Per docs/ui_flow_spec.md and docs/risk_copywriting_library.md
 */
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { assessScreening } from '../api/client'
import { AssessmentRequest, AssessmentResponse } from '../types'
import './QuickAssessment.css'

// Critical fields that should show warnings if missing
const CRITICAL_FIELDS = ['sleep_quality', 'average_screen_time', 'sleep_duration', 'stress_level']

// Field labels mapping
const FIELD_LABELS: Record<string, string> = {
  sleep_quality: 'Chất lượng ngủ',
  average_screen_time: 'Thời gian màn hình',
  sleep_duration: 'Thời lượng ngủ',
  stress_level: 'Mức độ căng thẳng',
}

export default function QuickAssessment() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  
  const [formData, setFormData] = useState<AssessmentRequest>(() => {
    // Load from localStorage if available
    const saved = localStorage.getItem('assessment_form_data')
    return saved ? JSON.parse(saved) : {}
  })

  // Save to localStorage on change
  useEffect(() => {
    localStorage.setItem('assessment_form_data', JSON.stringify(formData))
  }, [formData])

  const handleChange = (field: keyof AssessmentRequest, value: number | undefined) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    // Clear error when user types
    if (error) setError(null)
  }

  const validateAndWarn = (): boolean => {
    const missing: string[] = []
    CRITICAL_FIELDS.forEach((field) => {
      if (formData[field as keyof AssessmentRequest] === undefined || 
          formData[field as keyof AssessmentRequest] === null) {
        missing.push(FIELD_LABELS[field] || field)
      }
    })
    
    setWarnings(missing)
    // Always allow submit, just show warnings
    return true
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    // Validate and show warnings
    validateAndWarn()

    try {
      const result = await assessScreening(formData)
      
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

  return (
    <div className="quick-assessment-page">
      <header className="page-header">
        <h1>Sàng lọc nguy cơ</h1>
        <p className="subtitle">
          Điền thông tin để hệ thống đánh giá nguy cơ. Bạn có thể bỏ qua một số mục,
          nhưng kết quả sẽ chính xác hơn nếu điền đầy đủ.
        </p>
      </header>

      {warnings.length > 0 && (
        <div className="warning-banner">
          <strong>Lưu ý:</strong> Thiếu thông tin: {warnings.join(', ')}. 
          Kết quả sẽ kém chắc hơn nếu không điền đầy đủ.
        </div>
      )}

      <form onSubmit={handleSubmit} className="assessment-form">
        {/* Recommended Fields Section */}
        <section className="form-section">
          <h2>Thông tin quan trọng (khuyến nghị)</h2>
          <p className="section-tooltip">Điền các mục này để tăng độ chính xác</p>

          <div className="form-group">
            <label>
              Thời gian nhìn màn hình trung bình (giờ/ngày) *
              <input
                type="number"
                step="0.5"
                min="0"
                max="24"
                value={formData.average_screen_time ?? ''}
                onChange={(e) =>
                  handleChange(
                    'average_screen_time',
                    e.target.value ? parseFloat(e.target.value) : undefined
                  )
                }
                placeholder="Ví dụ: 8.5"
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Thời lượng ngủ (giờ/ngày) *
              <input
                type="number"
                step="0.5"
                min="0"
                max="24"
                value={formData.sleep_duration ?? ''}
                onChange={(e) =>
                  handleChange(
                    'sleep_duration',
                    e.target.value ? parseFloat(e.target.value) : undefined
                  )
                }
                placeholder="Ví dụ: 7.5"
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Chất lượng ngủ (1-5, 5 = tốt nhất) *
              <input
                type="number"
                min="1"
                max="5"
                value={formData.sleep_quality ?? ''}
                onChange={(e) =>
                  handleChange(
                    'sleep_quality',
                    e.target.value ? parseInt(e.target.value) : undefined
                  )
                }
                placeholder="1-5"
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Mức độ căng thẳng (1-5, 5 = rất cao) *
              <input
                type="number"
                min="1"
                max="5"
                value={formData.stress_level ?? ''}
                onChange={(e) =>
                  handleChange(
                    'stress_level',
                    e.target.value ? parseInt(e.target.value) : undefined
                  )
                }
                placeholder="1-5"
              />
            </label>
          </div>
        </section>

        {/* Optional Fields Section */}
        <section className="form-section">
          <h2>Thông tin bổ sung (tùy chọn)</h2>
          <p className="section-tooltip">Điền thêm để tăng độ chính xác</p>

          <div className="form-row">
            <div className="form-group">
              <label>
                Tuổi (18-45)
                <input
                  type="number"
                  min="18"
                  max="45"
                  value={formData.age ?? ''}
                  onChange={(e) =>
                    handleChange('age', e.target.value ? parseInt(e.target.value) : undefined)
                  }
                />
              </label>
            </div>

            <div className="form-group">
              <label>
                Giới tính
                <select
                  value={formData.gender ?? ''}
                  onChange={(e) =>
                    handleChange('gender', e.target.value ? parseInt(e.target.value) : undefined)
                  }
                >
                  <option value="">-- Chọn --</option>
                  <option value="0">Nữ</option>
                  <option value="1">Nam</option>
                </select>
              </label>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>
                Chiều cao (cm)
                <input
                  type="number"
                  min="120"
                  max="230"
                  value={formData.height ?? ''}
                  onChange={(e) =>
                    handleChange('height', e.target.value ? parseInt(e.target.value) : undefined)
                  }
                />
              </label>
            </div>

            <div className="form-group">
              <label>
                Cân nặng (kg)
                <input
                  type="number"
                  min="30"
                  max="250"
                  value={formData.weight ?? ''}
                  onChange={(e) =>
                    handleChange('weight', e.target.value ? parseInt(e.target.value) : undefined)
                  }
                />
              </label>
            </div>
          </div>

          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={formData.smart_device_before_bed === 1}
                onChange={(e) =>
                  handleChange('smart_device_before_bed', e.target.checked ? 1 : 0)
                }
              />
              Dùng thiết bị thông minh trước khi ngủ
            </label>
          </div>

          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={formData.bluelight_filter === 1}
                onChange={(e) =>
                  handleChange('bluelight_filter', e.target.checked ? 1 : 0)
                }
              />
              Sử dụng bộ lọc ánh sáng xanh
            </label>
          </div>

          <div className="form-group">
            <label>
              Số bước đi hàng ngày
              <input
                type="number"
                min="0"
                max="50000"
                value={formData.daily_steps ?? ''}
                onChange={(e) =>
                  handleChange('daily_steps', e.target.value ? parseInt(e.target.value) : undefined)
                }
              />
            </label>
          </div>

          <div className="form-group">
            <label>
              Nhịp tim (bpm)
              <input
                type="number"
                min="40"
                max="220"
                value={formData.heart_rate ?? ''}
                onChange={(e) =>
                  handleChange('heart_rate', e.target.value ? parseInt(e.target.value) : undefined)
                }
              />
            </label>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>
                Huyết áp tâm thu (mmHg)
                <input
                  type="number"
                  min="70"
                  max="250"
                  value={formData.systolic ?? ''}
                  onChange={(e) =>
                    handleChange('systolic', e.target.value ? parseInt(e.target.value) : undefined)
                  }
                />
              </label>
            </div>

            <div className="form-group">
              <label>
                Huyết áp tâm trương (mmHg)
                <input
                  type="number"
                  min="40"
                  max="150"
                  value={formData.diastolic ?? ''}
                  onChange={(e) =>
                    handleChange('diastolic', e.target.value ? parseInt(e.target.value) : undefined)
                  }
                />
              </label>
            </div>
          </div>
        </section>

        {error && <div className="error-message">{error}</div>}

        <div className="form-actions">
          <button type="submit" disabled={loading} className="submit-button primary">
            {loading ? 'Đang xử lý...' : 'Xem kết quả sàng lọc'}
          </button>
        </div>
      </form>
    </div>
  )
}
