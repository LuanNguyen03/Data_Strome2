/**
 * Result Page - Renders Assessment Response Contract
 * Per docs/output_contract.md and docs/risk_copywriting_library.md
 */
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AssessmentResponse, ModeUsed, RiskLevel, Confidence } from '../types'
import './Result.css'

export default function Result() {
  const navigate = useNavigate()
  const [response, setResponse] = useState<AssessmentResponse | null>(null)
  const [showTreatment, setShowTreatment] = useState(true)

  useEffect(() => {
    // Load response from localStorage
    const saved = localStorage.getItem('assessment_response')
    if (saved) {
      try {
        const parsedResponse = JSON.parse(saved)
        setResponse(parsedResponse)
        // Auto-show treatment if available
        if (parsedResponse.treatment_recommendations) {
          setShowTreatment(true)
        }
      } catch (e) {
        console.error('Failed to parse saved response:', e)
        // If no valid response, redirect to assessment
        navigate('/quick-assessment')
      }
    } else {
      // If no response, redirect to assessment
      navigate('/quick-assessment')
    }
  }, [navigate])

  if (!response) {
    return <div className="result-page loading">Đang tải kết quả...</div>
  }

  const getRiskLevelColor = (level: RiskLevel): string => {
    switch (level) {
      case RiskLevel.LOW:
        return 'low'
      case RiskLevel.MEDIUM:
        return 'medium'
      case RiskLevel.HIGH:
        return 'high'
      default:
        return 'unknown'
    }
  }

  const getConfidenceText = (confidence: Confidence): string => {
    switch (confidence) {
      case Confidence.HIGH:
        return 'Độ chắc: Cao — thông tin tương đối đầy đủ.'
      case Confidence.MEDIUM:
        return 'Độ chắc: Trung bình — thiếu một vài thông tin, kết quả có thể dao động.'
      case Confidence.LOW:
        return 'Độ chắc: Thấp — thiếu nhiều thông tin quan trọng. Bạn nên bổ sung để kết quả đáng tin hơn.'
      default:
        return ''
    }
  }

  const getModeLabel = (mode: ModeUsed): string => {
    switch (mode) {
      case ModeUsed.A_ONLY_SCREENING:
        return 'Sàng lọc (không triệu chứng)'
      case ModeUsed.B_WITH_SYMPTOMS:
        return 'Phân loại (có triệu chứng)'
      default:
        return ''
    }
  }

  const shouldShowSymptomCTA = 
    response.mode_used === ModeUsed.A_ONLY_SCREENING && 
    response.screening?.trigger_symptom === true

  return (
    <div className="result-page">
      <header className="page-header">
        <h1>Kết quả đánh giá</h1>
        <div className="mode-badge">
          {getModeLabel(response.mode_used)}
        </div>
      </header>

      {/* Main Score Display */}
      <section className="score-section">
        <div className="score-circle">
          <div className="score-value">{Math.round(response.risk_score)}</div>
          <div className="score-label">Điểm nguy cơ</div>
        </div>
        <div className={`risk-level-badge ${getRiskLevelColor(response.risk_level)}`}>
          {response.risk_level === RiskLevel.LOW && 'Nguy cơ thấp'}
          {response.risk_level === RiskLevel.MEDIUM && 'Nguy cơ trung bình'}
          {response.risk_level === RiskLevel.HIGH && 'Nguy cơ cao'}
        </div>
      </section>

      {/* Confidence Badge */}
      <section className="confidence-section">
        <div className={`confidence-badge ${response.confidence.toLowerCase()}`}>
          {getConfidenceText(response.confidence)}
        </div>
        {response.missing_fields.length > 0 && (
          <div className="missing-fields-warning">
            <strong>Thiếu thông tin:</strong> {response.missing_fields.join(', ')}.
            <br />
            Bạn có thể bổ sung để tăng độ chắc của kết quả.
          </div>
        )}
      </section>

      {/* Top Factors */}
      {response.top_factors.length > 0 && (
        <section className="factors-section">
          <h2>Các yếu tố ảnh hưởng</h2>
          <ul className="factors-list">
            {response.top_factors.map((factor, idx) => (
              <li key={idx} className={`factor-item ${factor.direction}`}>
                <span className="factor-strength">{factor.strength}</span>
                <span className="factor-note">{factor.note}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* AI Treatment Recommendations */}
      {response.treatment_recommendations && (
        <section className="treatment-section">
          <div className="treatment-header">
            <h2>Hướng điều trị đề xuất (AI)</h2>
            <button 
              className="toggle-button"
              onClick={() => setShowTreatment(!showTreatment)}
            >
              {showTreatment ? 'Ẩn' : 'Hiện'} khuyến nghị
            </button>
          </div>
          {showTreatment && (
            <div className="treatment-content">
              {response.treatment_recommendations.split('\n').map((line, idx) => {
                if (line.trim().startsWith('-') || line.trim().startsWith('*')) {
                  return <li key={idx} className="treatment-item">{line.trim().substring(1).trim()}</li>;
                }
                if (line.trim().match(/^\d+\./)) {
                  return <li key={idx} className="treatment-item">{line.trim().replace(/^\d+\.\s*/, '')}</li>;
                }
                return line.trim() ? <p key={idx}>{line}</p> : <br key={idx} />;
              })}
            </div>
          )}
        </section>
      )}

      {/* Next Steps */}
      <section className="next-steps-section">
        <h2>{response.next_step.title}</h2>
        <ul className="actions-list">
          {response.next_step.actions.map((action, idx) => (
            <li key={idx}>{action}</li>
          ))}
        </ul>
        {response.next_step.ask_for_more_info.length > 0 && (
          <div className="ask-for-more">
            <strong>Gợi ý bổ sung:</strong>{' '}
            {response.next_step.ask_for_more_info.join(', ')}
          </div>
        )}
      </section>

      {/* CTA to Symptoms (if trigger_symptom) */}
      {shouldShowSymptomCTA && (
        <section className="cta-section">
          <div className="cta-card">
            <h3>Bổ sung thông tin để phân loại rõ hơn</h3>
            <p>
              Trả lời thêm 3 triệu chứng mắt để hệ thống phân loại chính xác hơn.
            </p>
            <button
              onClick={() => navigate('/symptoms')}
              className="cta-button primary"
            >
              Trả lời thêm triệu chứng (30 giây)
            </button>
          </div>
        </section>
      )}

      {/* Disclaimers - ALWAYS SHOWN */}
      <section className="disclaimers-section">
        <h2>Lưu ý quan trọng</h2>
        <ul className="disclaimers-list">
          {response.disclaimers.map((disclaimer, idx) => (
            <li key={idx} className="disclaimer-item">
              {disclaimer}
            </li>
          ))}
        </ul>
      </section>

      {/* Action Buttons */}
      <section className="action-buttons">
        <button
          onClick={() => navigate('/quick-assessment')}
          className="button secondary"
        >
          Đánh giá lại
        </button>
        {response.mode_used === ModeUsed.A_ONLY_SCREENING && (
          <button
            onClick={() => navigate('/symptoms')}
            className="button secondary"
          >
            Thêm triệu chứng
          </button>
        )}
      </section>
    </div>
  )
}
