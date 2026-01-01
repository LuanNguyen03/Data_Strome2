import { useNavigate } from 'react-router-dom'
import { AssessmentResponse, RiskLevel, Confidence, ModeUsed } from '../types'
import './ResultPage.css'

interface Props {
  result: AssessmentResponse
}

export default function ResultPage({ result }: Props) {
  const navigate = useNavigate()

  const getRiskLevelColor = (level: RiskLevel) => {
    switch (level) {
      case RiskLevel.LOW:
        return '#28a745'
      case RiskLevel.MEDIUM:
        return '#ffc107'
      case RiskLevel.HIGH:
        return '#dc3545'
      default:
        return '#6c757d'
    }
  }

  const getConfidenceText = (confidence: Confidence) => {
    switch (confidence) {
      case Confidence.HIGH:
        return 'Cao — thông tin tương đối đầy đủ.'
      case Confidence.MEDIUM:
        return 'Trung bình — thiếu một vài thông tin, kết quả có thể dao động.'
      case Confidence.LOW:
        return 'Thấp — thiếu nhiều thông tin quan trọng. Bạn nên bổ sung để kết quả đáng tin hơn.'
      default:
        return ''
    }
  }

  const getModeLabel = (mode: ModeUsed) => {
    switch (mode) {
      case ModeUsed.A_ONLY_SCREENING:
        return 'Sàng lọc nguy cơ (không triệu chứng)'
      case ModeUsed.B_WITH_SYMPTOMS:
        return 'Phân loại khi có triệu chứng'
      default:
        return ''
    }
  }

  return (
    <div className="result-page">
      <div className="result-header">
        <h2>Kết quả đánh giá</h2>
        <span className="mode-badge">{getModeLabel(result.mode_used)}</span>
      </div>

      <div className="result-main">
        <div className="risk-score-card">
          <div className="score-circle" style={{ borderColor: getRiskLevelColor(result.risk_level) }}>
            <div className="score-value">{Math.round(result.risk_score)}</div>
            <div className="score-label">Điểm nguy cơ</div>
          </div>
          <div className="risk-level" style={{ color: getRiskLevelColor(result.risk_level) }}>
            {result.risk_level === RiskLevel.LOW && 'Nguy cơ thấp'}
            {result.risk_level === RiskLevel.MEDIUM && 'Nguy cơ trung bình'}
            {result.risk_level === RiskLevel.HIGH && 'Nguy cơ cao'}
          </div>
        </div>

        <div className="confidence-badge">
          <strong>Độ chắc: {result.confidence}</strong>
          <p>{getConfidenceText(result.confidence)}</p>
        </div>

        {result.missing_fields.length > 0 && (
          <div className="missing-fields">
            <h3>Thiếu thông tin</h3>
            <ul>
              {result.missing_fields.map((field) => (
                <li key={field}>{field}</li>
              ))}
            </ul>
            <p className="missing-note">
              Bạn có thể bổ sung để tăng độ chắc của kết quả.
            </p>
          </div>
        )}

        {result.top_factors.length > 0 && (
          <div className="top-factors">
            <h3>Các yếu tố ảnh hưởng</h3>
            <ul>
              {result.top_factors.map((factor, idx) => (
                <li key={idx}>
                  <strong>{factor.feature}:</strong> {factor.note}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="next-steps">
          <h3>{result.next_step.title}</h3>
          <ul>
            {result.next_step.actions.map((action, idx) => (
              <li key={idx}>{action}</li>
            ))}
          </ul>
          {result.screening?.trigger_symptom && (
            <div className="symptom-cta">
              <p>Bạn có muốn trả lời thêm về triệu chứng để phân loại rõ hơn không?</p>
              <button
                onClick={() => navigate('/')}
                className="cta-button"
              >
                Bổ sung triệu chứng
              </button>
            </div>
          )}
        </div>

        <div className="disclaimers">
          {result.disclaimers.map((disclaimer, idx) => (
            <p key={idx} className="disclaimer-text">
              {disclaimer}
            </p>
          ))}
        </div>
      </div>

      <button onClick={() => navigate('/')} className="back-button">
        Đánh giá lại
      </button>
    </div>
  )
}

