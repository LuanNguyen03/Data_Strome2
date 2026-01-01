/**
 * Dashboard Page - 3 Tabs: Overview, Risk Drivers, Symptom & Triage
 * Per docs/metrics_and_reporting.md and docs/02_olap_duckdb_plan.md
 */
import { useState, useEffect } from 'react'
import { getKPI, listKPIs } from '../api/client'
import './Dashboard.css'

type TabType = 'overview' | 'risk-drivers' | 'symptom-triage'

interface KPIData {
  name: string
  filename: string
  page: number
  page_size: number
  total_rows: number
  total_pages: number
  has_next: boolean
  has_prev: boolean
  data: any[]
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabType>('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // KPI data state
  const [ageGenderData, setAgeGenderData] = useState<KPIData | null>(null)
  const [screenSleepData, setScreenSleepData] = useState<KPIData | null>(null)
  const [symptomScoreData, setSymptomScoreData] = useState<KPIData | null>(null)
  const [stressSleepData, setStressSleepData] = useState<KPIData | null>(null)
  const [dataQualityData, setDataQualityData] = useState<KPIData | null>(null)

  useEffect(() => {
    loadAllKPIs()
  }, [])

  const loadAllKPIs = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Load all KPI datasets
      const [ageGender, screenSleep, symptomScore, stressSleep, dataQuality] = await Promise.all([
        getKPI('age_gender', 1, 1000).catch(() => null),
        getKPI('screen_sleep', 1, 1000).catch(() => null),
        getKPI('symptom_score', 1, 1000).catch(() => null),
        getKPI('stress_sleepband', 1, 1000).catch(() => null),
        getKPI('data_quality_group', 1, 1000).catch(() => null),
      ])

      if (ageGender) setAgeGenderData(ageGender)
      if (screenSleep) setScreenSleepData(screenSleep)
      if (symptomScore) setSymptomScoreData(symptomScore)
      if (stressSleep) setStressSleepData(stressSleep)
      if (dataQuality) setDataQualityData(dataQuality)
    } catch (err: any) {
      setError(err.message || 'Failed to load KPI data')
    } finally {
      setLoading(false)
    }
  }

  const formatRate = (rate: number | null | undefined): string => {
    if (rate === null || rate === undefined) return 'N/A'
    return `${(rate * 100).toFixed(1)}%`
  }

  const formatNumber = (n: number | null | undefined): string => {
    if (n === null || n === undefined) return 'N/A'
    return n.toLocaleString()
  }

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="loading-state">Đang tải dữ liệu...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="dashboard-page">
        <div className="error-state">
          <p>Lỗi: {error}</p>
          <button onClick={loadAllKPIs}>Thử lại</button>
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <h1>Dashboard Phân tích</h1>
        <p className="subtitle">Xu hướng và tương quan trong dataset</p>
      </header>

      {/* Tabs */}
      <div className="dashboard-tabs">
        <button
          className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Tổng quan
        </button>
        <button
          className={`tab-button ${activeTab === 'risk-drivers' ? 'active' : ''}`}
          onClick={() => setActiveTab('risk-drivers')}
        >
          Yếu tố nguy cơ
        </button>
        <button
          className={`tab-button ${activeTab === 'symptom-triage' ? 'active' : ''}`}
          onClick={() => setActiveTab('symptom-triage')}
        >
          Triệu chứng & Phân loại
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <OverviewTab
            ageGenderData={ageGenderData}
            dataQualityData={dataQualityData}
            formatRate={formatRate}
            formatNumber={formatNumber}
          />
        )}
        {activeTab === 'risk-drivers' && (
          <RiskDriversTab
            screenSleepData={screenSleepData}
            stressSleepData={stressSleepData}
            formatRate={formatRate}
            formatNumber={formatNumber}
          />
        )}
        {activeTab === 'symptom-triage' && (
          <SymptomTriageTab
            symptomScoreData={symptomScoreData}
            formatRate={formatRate}
            formatNumber={formatNumber}
          />
        )}
      </div>
    </div>
  )
}

// Overview Tab Component
function OverviewTab({
  ageGenderData,
  dataQualityData,
  formatRate,
  formatNumber,
}: {
  ageGenderData: KPIData | null
  dataQualityData: KPIData | null
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  // Calculate totals
  const totalRecords = ageGenderData?.data.reduce((sum, row) => sum + (row.n || 0), 0) || 0
  const totalPositives = ageGenderData?.data.reduce((sum, row) => sum + (row.positives || 0), 0) || 0
  const overallRate = totalRecords > 0 ? totalPositives / totalRecords : 0

  // Get top 5 missing rates from data quality
  const topMissingFields = dataQualityData?.data
    .flatMap((row) => [
      { field: 'screen_time', rate: row.missing_rate_screen_time || 0, n: row.n },
      { field: 'sleep_quality', rate: row.missing_rate_sleep_quality || 0, n: row.n },
      { field: 'bp', rate: row.missing_rate_bp || 0, n: row.n },
    ])
    .sort((a, b) => b.rate - a.rate)
    .slice(0, 5) || []

  const avgValidityRatio =
    dataQualityData?.data.reduce((sum, row) => sum + (row.avg_validity_ratio || 0), 0) /
      (dataQualityData?.data.length || 1) || 0

  return (
    <div className="overview-tab">
      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <h3>Tổng số bản ghi</h3>
          <div className="card-value">{formatNumber(totalRecords)}</div>
          <div className="card-label">n</div>
        </div>
        <div className="summary-card">
          <h3>Tỷ lệ DED dương tính</h3>
          <div className="card-value">{formatRate(overallRate)}</div>
          <div className="card-label">{formatNumber(totalPositives)} / {formatNumber(totalRecords)}</div>
        </div>
        <div className="summary-card">
          <h3>Tỷ lệ hợp lệ trung bình</h3>
          <div className="card-value">{formatRate(avgValidityRatio)}</div>
          <div className="card-label">validity_ratio</div>
        </div>
      </div>

      {/* DED Rate by Age Band */}
      {ageGenderData && (
        <div className="chart-section">
          <h2>Tỷ lệ DED theo nhóm tuổi và giới tính</h2>
          <p className="chart-subtitle">
            n = {formatNumber(totalRecords)} | Tỷ lệ tổng: {formatRate(overallRate)}
          </p>
          <AgeGenderChart data={ageGenderData.data} formatRate={formatRate} formatNumber={formatNumber} />
          <p className="chart-footnote">
            * Tương quan, không kết luận nhân quả
          </p>
        </div>
      )}

      {/* Data Quality */}
      {dataQualityData && (
        <div className="chart-section">
          <h2>Chất lượng dữ liệu</h2>
          <p className="chart-subtitle">Top 5 trường thiếu dữ liệu nhiều nhất</p>
          <DataQualityChart data={topMissingFields} formatRate={formatRate} formatNumber={formatNumber} />
          <p className="chart-footnote">
            * Tương quan, không kết luận nhân quả
          </p>
        </div>
      )}
    </div>
  )
}

// Risk Drivers Tab Component
function RiskDriversTab({
  screenSleepData,
  stressSleepData,
  formatRate,
  formatNumber,
}: {
  screenSleepData: KPIData | null
  stressSleepData: KPIData | null
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  return (
    <div className="risk-drivers-tab">
      {/* Screen Time × Sleep Quality Heatmap */}
      {screenSleepData && (
        <div className="chart-section">
          <h2>Heatmap: Thời gian màn hình × Chất lượng ngủ</h2>
          <p className="chart-subtitle">
            Tỷ lệ DED theo kết hợp screen_time_band và sleep_quality
          </p>
          <HeatmapTable
            data={screenSleepData.data}
            rowKey="screen_time_band"
            colKey="sleep_quality"
            formatRate={formatRate}
            formatNumber={formatNumber}
          />
          <p className="chart-footnote">
            * Tương quan, không kết luận nhân quả. Mỗi ô hiển thị: rate (n)
          </p>
        </div>
      )}

      {/* Stress Level × Sleep Duration Heatmap */}
      {stressSleepData && (
        <div className="chart-section">
          <h2>Heatmap: Mức độ căng thẳng × Thời lượng ngủ</h2>
          <p className="chart-subtitle">
            Tỷ lệ DED theo kết hợp stress_level và sleep_duration_band
          </p>
          <HeatmapTable
            data={stressSleepData.data}
            rowKey="stress_level"
            colKey="sleep_duration_band"
            formatRate={formatRate}
            formatNumber={formatNumber}
          />
          <p className="chart-footnote">
            * Tương quan, không kết luận nhân quả. Mỗi ô hiển thị: rate (n)
          </p>
        </div>
      )}

      {/* Bar Charts */}
      {screenSleepData && (
        <div className="chart-section">
          <h2>Tỷ lệ DED theo thời gian màn hình</h2>
          <p className="chart-subtitle">Nhóm theo screen_time_band</p>
          <BarChart
            data={aggregateByKey(screenSleepData.data, 'screen_time_band')}
            xKey="screen_time_band"
            formatRate={formatRate}
            formatNumber={formatNumber}
          />
          <p className="chart-footnote">
            * Tương quan, không kết luận nhân quả
          </p>
        </div>
      )}
    </div>
  )
}

// Symptom & Triage Tab Component
function SymptomTriageTab({
  symptomScoreData,
  formatRate,
  formatNumber,
}: {
  symptomScoreData: KPIData | null
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  return (
    <div className="symptom-triage-tab">
      {symptomScoreData && symptomScoreData.data.length > 0 ? (
        <>
          <div className="chart-section">
            <h2>Tỷ lệ DED theo số triệu chứng</h2>
            <p className="chart-subtitle">
              symptom_score từ 0 (không triệu chứng) đến 3 (3 triệu chứng)
            </p>
            <BarChart
              data={symptomScoreData.data}
              xKey="symptom_score"
              formatRate={formatRate}
              formatNumber={formatNumber}
            />
            <p className="chart-footnote">
              * Tương quan, không kết luận nhân quả. Symptom gần nhãn → triage mạnh hơn, không dùng cho dự báo sớm.
            </p>
          </div>

          <div className="chart-section">
            <h2>Bảng chi tiết: Symptom Score</h2>
            <SymptomScoreTable data={symptomScoreData.data} formatRate={formatRate} formatNumber={formatNumber} />
            <p className="chart-footnote">
              * Tương quan, không kết luận nhân quả
            </p>
          </div>
        </>
      ) : (
        <div className="no-data">
          <p>Chưa có dữ liệu triệu chứng. Vui lòng chạy OLAP generation để tạo KPI dataset.</p>
        </div>
      )}
    </div>
  )
}

// Helper Components

function AgeGenderChart({
  data,
  formatRate,
  formatNumber,
}: {
  data: any[]
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  // Group by age_band and create bars
  const ageBands = [...new Set(data.map((r) => r.age_band).filter(Boolean))].sort()
  const genders = ['0', '1'] // 0=F, 1=M

  return (
    <div className="age-gender-chart">
      <div className="bar-chart-container">
        {ageBands.map((ageBand) => {
          const rows = data.filter((r) => r.age_band === ageBand)
          const femaleRow = rows.find((r) => r.gender === 0 || r.gender === '0')
          const maleRow = rows.find((r) => r.gender === 1 || r.gender === '1')

          return (
            <div key={ageBand} className="bar-group">
              <div className="bar-label">{ageBand}</div>
              <div className="bars">
                <div className="bar-item">
                  <div className="bar-label-small">Nữ</div>
                  <div className="bar-wrapper">
                    <div
                      className="bar female"
                      style={{
                        width: `${((femaleRow?.rate || 0) * 100).toFixed(1)}%`,
                      }}
                    >
                      <span className="bar-value">
                        {formatRate(femaleRow?.rate)} (n={formatNumber(femaleRow?.n)})
                      </span>
                    </div>
                  </div>
                </div>
                <div className="bar-item">
                  <div className="bar-label-small">Nam</div>
                  <div className="bar-wrapper">
                    <div
                      className="bar male"
                      style={{
                        width: `${((maleRow?.rate || 0) * 100).toFixed(1)}%`,
                      }}
                    >
                      <span className="bar-value">
                        {formatRate(maleRow?.rate)} (n={formatNumber(maleRow?.n)})
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function HeatmapTable({
  data,
  rowKey,
  colKey,
  formatRate,
  formatNumber,
}: {
  data: any[]
  rowKey: string
  colKey: string
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  const rows = [...new Set(data.map((r) => r[rowKey]).filter(Boolean))].sort()
  const cols = [...new Set(data.map((r) => r[colKey]).filter(Boolean))].sort()

  const getCell = (rowVal: any, colVal: any) => {
    return data.find((r) => r[rowKey] === rowVal && r[colKey] === colVal)
  }

  const getMaxRate = () => {
    return Math.max(...data.map((r) => r.rate || 0))
  }

  const maxRate = getMaxRate()

  return (
    <div className="heatmap-container">
      <table className="heatmap-table">
        <thead>
          <tr>
            <th>{rowKey}</th>
            {cols.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const rowData = data.filter((r) => r[rowKey] === row)
            const rowTotal = rowData.reduce((sum, r) => sum + (r.n || 0), 0)

            return (
              <tr key={row}>
                <td className="row-header">
                  {row}
                  <span className="row-total">(n={formatNumber(rowTotal)})</span>
                </td>
                {cols.map((col) => {
                  const cell = getCell(row, col)
                  const rate = cell?.rate || 0
                  const n = cell?.n || 0
                  const intensity = maxRate > 0 ? rate / maxRate : 0

                  return (
                    <td
                      key={col}
                      className="heatmap-cell"
                      style={{
                        backgroundColor: `rgba(220, 53, 69, ${intensity * 0.7 + 0.1})`,
                        color: intensity > 0.5 ? 'white' : 'black',
                      }}
                      title={`${rowKey}=${row}, ${colKey}=${col}: ${formatRate(rate)} (n=${formatNumber(n)})`}
                    >
                      <div className="cell-rate">{formatRate(rate)}</div>
                      <div className="cell-n">n={formatNumber(n)}</div>
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function BarChart({
  data,
  xKey,
  formatRate,
  formatNumber,
}: {
  data: any[]
  xKey: string
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  const maxRate = Math.max(...data.map((r) => r.rate || 0))

  return (
    <div className="bar-chart-container">
      {data.map((row) => {
        const rate = row.rate || 0
        const n = row.n || 0
        const width = maxRate > 0 ? `${(rate / maxRate) * 100}%` : '0%'

        return (
          <div key={row[xKey]} className="bar-group-simple">
            <div className="bar-label">{row[xKey]}</div>
            <div className="bar-wrapper">
              <div className="bar simple" style={{ width }}>
                <span className="bar-value">
                  {formatRate(rate)} (n={formatNumber(n)})
                </span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function DataQualityChart({
  data,
  formatRate,
  formatNumber,
}: {
  data: { field: string; rate: number; n: number }[]
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  const maxRate = Math.max(...data.map((d) => d.rate))

  return (
    <div className="bar-chart-container">
      {data.map((item) => {
        const width = maxRate > 0 ? `${(item.rate / maxRate) * 100}%` : '0%'

        return (
          <div key={item.field} className="bar-group-simple">
            <div className="bar-label">{item.field}</div>
            <div className="bar-wrapper">
              <div className="bar quality" style={{ width }}>
                <span className="bar-value">
                  {formatRate(item.rate)} (n={formatNumber(item.n)})
                </span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function SymptomScoreTable({
  data,
  formatRate,
  formatNumber,
}: {
  data: any[]
  formatRate: (rate: number | null | undefined) => string
  formatNumber: (n: number | null | undefined) => string
}) {
  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Symptom Score</th>
            <th>Số lượng (n)</th>
            <th>Số dương tính</th>
            <th>Tỷ lệ DED</th>
          </tr>
        </thead>
        <tbody>
          {data
            .sort((a, b) => (a.symptom_score || 0) - (b.symptom_score || 0))
            .map((row) => (
              <tr key={row.symptom_score}>
                <td>{row.symptom_score}</td>
                <td>{formatNumber(row.n)}</td>
                <td>{formatNumber(row.positives)}</td>
                <td className="rate-cell">{formatRate(row.rate)}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}

// Helper function
function aggregateByKey(data: any[], key: string): any[] {
  const grouped = data.reduce((acc, row) => {
    const val = row[key]
    if (!acc[val]) {
      acc[val] = { [key]: val, n: 0, positives: 0 }
    }
    acc[val].n += row.n || 0
    acc[val].positives += row.positives || 0
    return acc
  }, {} as Record<string, any>)

  return Object.values(grouped).map((row: any) => ({
    ...row,
    rate: row.n > 0 ? row.positives / row.n : 0,
  }))
}
