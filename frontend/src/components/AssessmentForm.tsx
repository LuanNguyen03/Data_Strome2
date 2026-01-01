import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { assessRisk } from '../api/client'
import { AssessmentRequest } from '../types'
import './AssessmentForm.css'

interface Props {
  onResult: (result: any) => void
}

export default function AssessmentForm({ onResult }: Props) {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [formData, setFormData] = useState<AssessmentRequest>({
    age: undefined,
    gender: undefined,
    sleep_duration: undefined,
    sleep_quality: undefined,
    average_screen_time: undefined,
    stress_level: undefined,
  })

  const handleChange = (field: keyof AssessmentRequest, value: number | undefined) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const result = await assessRisk(formData)
      onResult(result)
      navigate('/result')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Có lỗi xảy ra. Vui lòng thử lại.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="assessment-form">
      <h2>Đánh giá nguy cơ khô mắt</h2>
      <p className="form-intro">
        Vui lòng điền thông tin để hệ thống đánh giá nguy cơ. Bạn có thể bỏ qua một số mục,
        nhưng kết quả sẽ chính xác hơn nếu điền đầy đủ.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Thông tin cơ bản</h3>
          <div className="form-group">
            <label>
              Tuổi (18-45)
              <input
                type="number"
                min="18"
                max="45"
                value={formData.age || ''}
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

        <div className="form-section">
          <h3>Giấc ngủ</h3>
          <div className="form-group">
            <label>
              Thời lượng ngủ (giờ/ngày)
              <input
                type="number"
                step="0.5"
                min="0"
                max="24"
                value={formData.sleep_duration || ''}
                onChange={(e) =>
                  handleChange(
                    'sleep_duration',
                    e.target.value ? parseFloat(e.target.value) : undefined
                  )
                }
              />
            </label>
          </div>
          <div className="form-group">
            <label>
              Chất lượng ngủ (1-5, 5 = tốt nhất)
              <input
                type="number"
                min="1"
                max="5"
                value={formData.sleep_quality || ''}
                onChange={(e) =>
                  handleChange(
                    'sleep_quality',
                    e.target.value ? parseInt(e.target.value) : undefined
                  )
                }
              />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3>Thời gian màn hình</h3>
          <div className="form-group">
            <label>
              Thời gian nhìn màn hình trung bình (giờ/ngày)
              <input
                type="number"
                step="0.5"
                min="0"
                max="24"
                value={formData.average_screen_time || ''}
                onChange={(e) =>
                  handleChange(
                    'average_screen_time',
                    e.target.value ? parseFloat(e.target.value) : undefined
                  )
                }
              />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3>Căng thẳng</h3>
          <div className="form-group">
            <label>
              Mức độ căng thẳng (1-5, 5 = rất cao)
              <input
                type="number"
                min="1"
                max="5"
                value={formData.stress_level || ''}
                onChange={(e) =>
                  handleChange(
                    'stress_level',
                    e.target.value ? parseInt(e.target.value) : undefined
                  )
                }
              />
            </label>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Đang xử lý...' : 'Xem kết quả sàng lọc'}
        </button>
      </form>
    </div>
  )
}

