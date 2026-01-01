import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import QuickAssessment from './pages/QuickAssessment'
import Symptoms from './pages/Symptoms'
import Result from './pages/Result'
import Dashboard from './pages/Dashboard'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <div className="header-content">
            <div>
              <h1>Dry Eye Disease Risk Assessment</h1>
              <p className="subtitle">Hệ thống hỗ trợ sàng lọc và phân loại nguy cơ</p>
            </div>
            <nav className="header-nav">
              <a href="/quick-assessment">Đánh giá</a>
              <a href="/dashboard">Dashboard</a>
            </nav>
          </div>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Navigate to="/quick-assessment" replace />} />
            <Route path="/quick-assessment" element={<QuickAssessment />} />
            <Route path="/symptoms" element={<Symptoms />} />
            <Route path="/result" element={<Result />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
        <footer className="app-footer">
          <p className="disclaimer">
            Kết quả chỉ hỗ trợ sàng lọc/triage, không thay thế chẩn đoán.
            Nếu triệu chứng kéo dài hoặc nặng, nên tham khảo bác sĩ.
          </p>
        </footer>
      </div>
    </BrowserRouter>
  )
}

export default App
